import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .progress import Progress

N_FOLDS = 5


def layout(root):
    root = Path(root)
    info = json.loads((root / "meta.json").read_text())
    return {"channels": [str(c) for c in info["ch_names"]], "sfreq": float(info["sfreq"]),
            "n_times": int(np.load(root / "test_X.npy", mmap_mode="r").shape[2])}


class EEGData:
    def __init__(self, root, channels=None, n_times=None, pool=True):
        self.root = Path(root)
        info = json.loads((self.root / "meta.json").read_text())
        stored = [str(c) for c in info["ch_names"]]
        self.sfreq = float(info["sfreq"])
        self.channels = list(stored) if channels is None else [str(c) for c in channels]
        lower = [c.lower() for c in stored]
        absent = [c for c in self.channels if c.lower() not in lower]
        assert not absent, f"channels absent from {self.root}: {absent}"
        self.pick = np.array([lower.index(c.lower()) for c in self.channels])
        shape = np.load(self.root / "test_X.npy", mmap_mode="r").shape
        self.n_times = int(shape[2]) if n_times is None else int(n_times)
        assert shape[2] >= self.n_times, f"{self.root} holds {shape[2]} samples per trial, {self.n_times} are needed"
        self.targets = np.load(self.root / "targets_dinov2_giant.npy").astype(np.float32)
        self.d_out = int(self.targets.shape[1])
        self.unit_targets = F.normalize(torch.from_numpy(self.targets), dim=1)
        self.pool_subject = np.concatenate([np.load(self.root / f"{s}_subject.npy") for s in ("train", "val")]).astype(np.int64)
        self.pool_image = np.concatenate([np.load(self.root / f"{s}_image.npy") for s in ("train", "val")]).astype(np.int64)
        self.test_subject = np.load(self.root / "test_subject.npy").astype(np.int64)
        self.test_image = np.load(self.root / "test_image.npy").astype(np.int64)
        self.subjects = np.unique(self.pool_subject)
        self.folds = np.array_split(self.subjects, N_FOLDS)
        self.gallery_pool, self.gallery_test = np.unique(self.pool_image), np.unique(self.test_image)
        self.gallery_all = np.union1d(self.gallery_pool, self.gallery_test)
        assert not set(self.gallery_test.tolist()) & set(self.gallery_pool.tolist()), "test images also appear in training"
        assert np.array_equal(np.unique(self.test_subject), self.subjects), "test and training people differ"
        t0 = time.time()
        self.test_x = self.read(["test"], "test trials")
        self.pool_x = self.read(["train", "val"], "training trials") if pool else None
        assert len(self.test_x) == len(self.test_subject) and (self.pool_x is None or len(self.pool_x) == len(self.pool_subject))
        print(f"{self.root.name} ready in {time.time() - t0:.0f}s | {len(self.pool_subject):,} training and {len(self.test_subject):,} "
              f"test trials | {len(self.subjects)} people | {len(self.channels)} channels x {self.n_times} samples at "
              f"{self.sfreq:g} Hz | {len(self.gallery_pool):,} training and {len(self.gallery_test):,} test images", flush=True)

    def read(self, splits, label):
        files = [np.load(self.root / f"{s}_X.npy", mmap_mode="r") for s in splits]
        X = np.empty((sum(len(f) for f in files), len(self.pick), self.n_times), np.float16)
        whole = files[0].shape[1] == len(self.pick) and np.array_equal(self.pick, np.arange(len(self.pick)))
        hb = Progress(f"loading {self.root.name} {label}", len(X))
        base = 0
        for f in files:
            for a in range(0, len(f), 65536):
                b = min(a + 65536, len(f))
                block = f[a:b, :, :self.n_times]
                X[base + a:base + b] = block if whole else block[:, self.pick]
                hb(b - a)
            base += len(f)
        return X

    def split(self, fold):
        if fold == "all":
            return self.subjects, np.arange(len(self.pool_subject)), np.arange(len(self.test_subject))
        held = self.folds[fold]
        return held, np.flatnonzero(~np.isin(self.pool_subject, held)), np.flatnonzero(np.isin(self.test_subject, held))
