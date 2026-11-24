import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import braindecode.models as BDM

from . import augment
from .data import N_FOLDS
from .inference import EEGEncoder, one_patch_per_channel
from .metrics import retrieval_scores
from .progress import Progress, release

PRETRAINED = "braindecode/labram-pretrained"
PUBLISHED = ("labram", "random")
COSINE = ("centered", "uncentered", "none")


def lr_factor(total, warmup):
    warm = max(1, min(warmup, total // 10))
    return lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))


def encoder_config(labram):
    config = json.loads(json.dumps(labram.get_config()))
    if isinstance(getattr(labram, "n_outputs", None), int):
        config["n_outputs"] = labram.n_outputs
    return config


def autocast(device):
    on = device.startswith("cuda")
    return torch.autocast(device_type="cuda" if on else "cpu", dtype=torch.float16, enabled=on)


@torch.no_grad()
def predict(model, X, rows, device, batch=1024, hb=None):
    out = []
    for a in range(0, len(rows), batch):
        x = torch.from_numpy(X[rows[a:a + batch]].astype(np.float32)).to(device)
        with autocast(device):
            out.append(model(x).float())
        if hb is not None:
            hb(len(x))
    return torch.cat(out)


class Session:
    def __init__(self, data, settings, out, hours=None):
        self.data, self.settings, self.out = data, dict(settings), Path(out)
        (self.out / "folds").mkdir(parents=True, exist_ok=True)
        self.deadline = None if hours is None else time.time() + hours * 3600
        self.build_lock, self.csv_lock, self.map_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.maps, self.tensors = {}, {}
        self.pool = torch.from_numpy(data.pool_x)
        people = data.subjects[:self.settings["monitor_people"]]
        self.monitor = np.flatnonzero(np.isin(data.test_subject, people))

    def final_files(self, name):
        return self.out / f"{name}.pt", self.out / f"{name}.json"

    def fold_file(self, name, fold):
        return self.out / "folds" / f"{name}_fold{fold}.npy"

    def done(self, job):
        name, fold = job
        if fold == "all":
            return all(p.exists() for p in self.final_files(name))
        return self.fold_file(name, fold).exists()

    def job_hours(self, fold):
        return 0.06 + self.settings["epochs"] * self.settings["epoch_hours"] * (1.1 if fold == "all" else 1.0)

    def append(self, file, row):
        with self.csv_lock:
            path = self.out / file
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def on(self, device):
        if device not in self.tensors:
            unit = self.data.unit_targets.to(device)
            self.tensors[device] = (unit, unit[torch.from_numpy(self.data.gallery_pool).to(device)].mean(0))
        return self.tensors[device]

    def transport_maps(self, fold, people):
        with self.map_lock:
            if fold not in self.maps:
                self.maps[fold] = augment.transport_maps(self.data.pool_x, self.data.pool_subject, people,
                                                         self.settings["covariance_trials"],
                                                         77 + (N_FOLDS if fold == "all" else fold))
            return self.maps[fold]

    def new_model(self, start):
        data = self.data
        if start == "labram":
            labram = BDM.Labram.from_pretrained(PRETRAINED)
        elif start == "random":
            labram = BDM.Labram(n_times=3000, n_chans=128, n_outputs=2)
        else:
            info = json.loads(self.final_files(start)[1].read_text())
            same = [c.lower() for c in info["channels"]] == [c.lower() for c in data.channels]
            assert same and info["n_times"] == data.n_times and info["d_out"] == data.d_out, \
                f"{start} was trained on another montage, window or embedding size"
            labram = BDM.Labram.from_config(dict(info["encoder_config"]))
        labram = one_patch_per_channel(labram)
        model = EEGEncoder(labram, data.channels, data.n_times, data.sfreq, data.d_out)
        if start not in PUBLISHED:
            model.load_state_dict(torch.load(self.final_files(start)[0], map_location="cpu"))
        return model, encoder_config(labram)

    def score(self, model, rows, device):
        model.eval()
        pred = predict(model, self.data.test_x, rows, device)
        unit = F.normalize(pred, dim=1).cpu().numpy()
        return pred, retrieval_scores(unit, self.data.test_subject[rows], self.data.test_image[rows], self.data.targets, device)

    def step(self, model, x, images, opt, scaler, logit_scale, params, cosine, device):
        cfg = self.settings
        unit, mean_unit = self.on(device)
        rows = torch.from_numpy(images).to(device)
        with autocast(device):
            p = model(x)
        p = F.normalize(p.float(), dim=1)
        scale = logit_scale.exp().clamp(max=100)
        t = unit[rows]
        uniq, inv = torch.unique(rows, return_inverse=True)
        loss = F.cross_entropy(scale * p @ unit[uniq].T, inv)
        if cosine == "uncentered":
            loss = loss + cfg["cosine_weight"] * (1 - (p * t).sum(1)).mean()
        elif cosine == "centered":
            loss = loss + cfg["cosine_weight"] * (1 - (p * F.normalize(t - mean_unit, dim=1)).sum(1)).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        return loss.detach()

    def train(self, spec, fold, device):
        cfg, data = self.settings, self.data
        name = spec["name"]
        held, tr, te = data.split(fold)
        tag = f"{name} {'all people' if fold == 'all' else f'fold {fold}'} on {device}"
        t_run = time.time()
        seed = spec["seed"] + (0 if fold == "all" else fold)
        rng = np.random.default_rng(seed)
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        with self.build_lock:
            torch.default_generator.manual_seed(seed)
            model, enc_cfg = self.new_model(spec["start"])
        model = model.to(device)
        people = np.unique(data.pool_subject[tr])
        maps = person_of = gen = same_image = None
        if spec["blend"]:
            same_image = augment.SameImageIndex(data.pool_image, tr)
        if spec["transport"] or spec["blend"]:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        if spec["transport"]:
            maps = self.transport_maps(fold, people).to(device)
            person_of = np.full(int(data.subjects.max()) + 1, -1, np.int64)
            person_of[people] = np.arange(len(people))
        logit_scale = nn.Parameter(torch.tensor(math.log(1 / cfg["temperature"]), device=device))
        bs, n_ep = cfg["batch_size"], cfg["epochs"]
        steps = max(1, min(cfg["samples_per_epoch"], len(tr)) // bs)
        monitor = self.monitor if fold == "all" else te
        groups = [{"params": list(model.head.parameters()) + [logit_scale], "lr": cfg["lr_head"]},
                  {"params": list(model.encoder.parameters()), "lr": cfg["lr_encoder"]}]
        params = [p for g in groups for p in g["params"]]
        opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(n_ep * steps, cfg["warmup_steps"]))
        scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
        print(f"[{tag}] start {spec['start']} | cosine term {spec['cosine']} | transport {spec['transport']} | blend {spec['blend']} | "
              f"{len(people)} training people, {len(tr):,} trials, {steps * bs:,} per epoch | {n_ep} epochs | seed {seed}", flush=True)
        for ep in range(n_ep):
            if self.deadline is not None and time.time() > self.deadline:
                print(f"[{tag}] time limit reached: stopped, nothing saved", flush=True)
                return None
            model.train()
            t0 = time.time()
            run_loss = torch.zeros((), device=device)
            perm = rng.permutation(tr)[:steps * bs]
            hb = Progress(f"{tag} epoch {ep + 1}/{n_ep}", steps)
            for b in range(steps):
                rows = perm[b * bs:(b + 1) * bs]
                x = self.pool[torch.from_numpy(rows)].to(device, non_blocking=True).float()
                if same_image is not None:
                    other = same_image.draw(data.pool_image[rows], rng)
                    x = augment.blend_same_image(x, self.pool[torch.from_numpy(other)].to(device).float(), gen,
                                                 cfg["blend_p"], cfg["blend_max"])
                if maps is not None:
                    x = augment.person_transport(x, torch.from_numpy(person_of[data.pool_subject[rows]]).to(device), maps, gen,
                                                 cfg["transport_p"])
                    x = augment.time_shift(x, gen, cfg["shift_p"], cfg["shift_max"])
                run_loss += self.step(model, x, data.pool_image[rows], opt, scaler, logit_scale, params, spec["cosine"], device)
                sched.step()
                hb()
            score = self.score(model, monitor, device)[1]
            temperature = 1 / logit_scale.exp().clamp(max=100).item()
            self.append("history.csv", {"model": name, "fold": str(fold), "epoch": ep + 1, "train_loss": run_loss.item() / steps,
                                        "top5_averaged": score["top5_averaged"], "top5_trial": score["top5_trial"],
                                        "temperature": temperature, "epoch_min": (time.time() - t0) / 60})
            print(f"[{tag}] epoch {ep + 1} | loss {run_loss.item() / steps:.3f} | {'seen' if fold == 'all' else 'held-out'} people, "
                  f"{len(data.gallery_test)} test images: top-5 {score['top5_averaged']:.4f}, single-trial top-5 "
                  f"{score['top5_trial']:.4f} | temperature {temperature:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        pred, score = self.score(model, te, device)
        row = {"model": name, "fold": str(fold), "epochs": n_ep, "top5_averaged": score["top5_averaged"],
               "top5_trial": score["top5_trial"], "hours": (time.time() - t_run) / 3600}
        if fold == "all":
            weights, info = self.final_files(name)
            info.write_text(json.dumps({"name": name, "encoder_config": enc_cfg, "channels": data.channels, "n_times": data.n_times,
                                        "sfreq": data.sfreq, "d_out": data.d_out, "spec": spec, "training": cfg,
                                        "data": data.root.name, "people": int(len(people)), "trials": int(len(tr))}, indent=1))
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"{weights}.tmp")
            os.replace(f"{weights}.tmp", weights)
        else:
            file = self.fold_file(name, fold)
            np.save(f"{file}.tmp.npy", pred.cpu().numpy().astype(np.float16))
            os.replace(f"{file}.tmp.npy", file)
        self.append("summary.csv", row)
        print(f"[{tag}] done in {row['hours']:.2f} h | {'seen' if fold == 'all' else 'held-out'} people, "
              f"{len(data.gallery_test)} test images: top-5 {row['top5_averaged']:.4f}, single-trial top-5 {row['top5_trial']:.4f}",
              flush=True)
        return row

    def run(self, specs, jobs, devices):
        todo = [j for j in jobs if not self.done(j)]
        state = {j: "pending" for j in todo}
        lock = threading.Lock()

        def next_job():
            with lock:
                for j in todo:
                    if state[j] != "pending":
                        continue
                    if self.deadline is not None and time.time() + 3600 * self.job_hours(j[1]) > self.deadline:
                        state[j] = "left"
                        print(f"[{j[0]} {j[1]}] not started: about {self.job_hours(j[1]):.1f} h needed, not enough time left",
                              flush=True)
                        continue
                    state[j] = "running"
                    return j
                return None

        def lane(device):
            while True:
                job = next_job()
                if job is None:
                    return
                state[job] = "done" if self.train(specs[job[0]], job[1], device) is not None else "left"
                release(device)

        print(f"{len(jobs) - len(todo)} jobs already finished, {len(todo)} to run on {devices} | about "
              f"{sum(self.job_hours(j[1]) for j in todo) / len(devices):.1f} h", flush=True)
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            for f in [ex.submit(lane, d) for d in devices]:
                f.result()
        return [j for j in jobs if not self.done(j)]


def check_specs(specs):
    names = [s["name"] for s in specs]
    assert len(set(names)) == len(names), "two models share a name"
    for s in specs:
        assert s["cosine"] in COSINE, f"{s['name']}: cosine must be one of {COSINE}"
        assert isinstance(s["transport"], bool) and isinstance(s["blend"], bool) and isinstance(s["seed"], int), s["name"]
