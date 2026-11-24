from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import braindecode.models as BDM

ENCODER_RATE = 200
ENCODER_SCALE = 1.349 * 0.1


def to_encoder_input(x, sfreq):
    n_out = int(x.shape[-1] * ENCODER_RATE / sfreq + 0.5)
    if n_out != x.shape[-1]:
        x = AF.resample(x, orig_freq=int(sfreq), new_freq=ENCODER_RATE)[..., :n_out]
    return x * ENCODER_SCALE


def one_patch_per_channel(labram):
    labram.patch_embed[0].n_patchs = 1
    table = labram._parameters.pop("temporal_embedding", None)
    if table is not None:
        labram.register_buffer("temporal_embedding", table.detach()[:, :2].clone())
    return labram


class TokenHead(nn.Module):
    def __init__(self, n_tokens, width, d_out, d_token, dropout=0.3):
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(width, d_token), nn.GELU())
        self.out = nn.Sequential(nn.LayerNorm(n_tokens * d_token), nn.Dropout(dropout), nn.Linear(n_tokens * d_token, d_out))

    def forward(self, tokens):
        return self.out(self.reduce(tokens).flatten(1))


class EEGEncoder(nn.Module):
    def __init__(self, labram, ch_names, n_times, sfreq, d_out):
        super().__init__()
        self.encoder = labram
        self.ch_names, self.sfreq = list(ch_names), float(sfreq)
        with torch.no_grad():
            tokens = self.tokens(torch.zeros(2, len(self.ch_names), n_times))
        self.head = TokenHead(tokens.shape[1], tokens.shape[2], d_out, d_token=max(8, 2048 // tokens.shape[1]))

    def tokens(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = to_encoder_input(x.float(), self.sfreq)
        return self.encoder(x, return_features=True, ch_names=self.ch_names)["features"]

    def forward(self, x):
        return self.head(self.tokens(x))


def build_member(encoder_config, weights, ch_names, n_times, sfreq, d_out):
    labram = one_patch_per_channel(BDM.Labram.from_config(dict(encoder_config)))
    model = EEGEncoder(labram, ch_names, n_times, sfreq, d_out)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()})
    return model.eval()


def fuse(outputs):
    return F.normalize(torch.stack([F.normalize(o.float(), dim=1) for o in outputs]).mean(0), dim=1)


class Ensemble(nn.Module):
    def __init__(self, members, info, meta):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.n_times, self.n_in, self.d_out = int(info["n_times"]), int(meta["n_chans"]), int(info["d_out"])
        names = [str(c).lower() for c in meta["ch_names"]]
        if len(names) != self.n_in or len(set(names)) != len(names):
            raise RuntimeError(f"unexpected channel list: {list(meta['ch_names'])}")
        missing = [c for c in info["channels"] if c.lower() not in names]
        if missing:
            raise RuntimeError(f"training channels absent from this montage: {missing}")
        if abs(float(meta["sfreq"]) - float(info["sfreq"])) > 1e-6:
            raise RuntimeError(f"sampling rate {meta['sfreq']} instead of {info['sfreq']}")
        if int(meta["n_times"]) < self.n_times:
            raise RuntimeError(f"window of {meta['n_times']} samples, shorter than the {self.n_times} used in training")
        if int(meta["n_outputs"]) != self.d_out:
            raise RuntimeError(f"embedding size {meta['n_outputs']} instead of {self.d_out}")
        self.register_buffer("index", torch.tensor([names.index(c.lower()) for c in info["channels"]]))

    def select(self, X):
        X = torch.as_tensor(X)
        if X.dim() != 3 or X.shape[1] != self.n_in or X.shape[2] < self.n_times:
            raise RuntimeError(f"unexpected input shape {tuple(X.shape)}")
        return X.float()[:, self.index, :self.n_times]

    def member_outputs(self, Z, batch=512):
        outs = [[] for _ in self.members]
        for a in range(0, Z.shape[0], batch):
            z = Z[a:a + batch]
            for k, member in enumerate(self.members):
                with torch.autocast(device_type=z.device.type, dtype=torch.float16, enabled=z.is_cuda):
                    outs[k].append(member(z).float())
        return [torch.cat(o) for o in outs]

    @torch.no_grad()
    def predict(self, X):
        Z = self.select(X)
        if Z.shape[0] == 0:
            return Z.new_zeros(0, self.d_out)
        return fuse(self.member_outputs(Z))


def self_test(model, root, threshold):
    ref = np.load(Path(root) / "reference.npz")
    with torch.no_grad():
        outs = model.member_outputs(torch.from_numpy(ref["x"]).to(model.index.device))
        got = [o.float().cpu() for o in outs] + [fuse(outs).cpu()]
    want = [torch.from_numpy(y) for y in ref["members"]] + [torch.from_numpy(ref["ensemble"])]
    if len(got) != len(want):
        raise RuntimeError(f"self-test failed: {len(got) - 1} models for {len(want) - 1} stored outputs")
    cos = min(F.cosine_similarity(g, w, dim=1).min().item() for g, w in zip(got, want))
    if not cos >= threshold:
        raise RuntimeError(f"self-test failed: cosine {cos:.6f} to the stored outputs")
    return cos


def load_ensemble(root, meta):
    root = Path(root)
    info = json.loads((root / "model.json").read_text())
    members = [build_member(m["encoder_config"], root / m["file"], info["channels"], info["n_times"], info["sfreq"], info["d_out"])
               for m in info["members"]]
    model = Ensemble(members, info, meta).to(meta["device"]).eval()
    self_test(model, root, info["self_test_cosine"])
    return model
