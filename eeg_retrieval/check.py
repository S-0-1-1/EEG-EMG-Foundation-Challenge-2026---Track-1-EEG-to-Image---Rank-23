import argparse
import importlib.util
import shutil
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import EEGData
from .metrics import retrieval_scores
from .progress import Progress, available_devices, run_on_devices

TOLERANCE = 1e-4
BATCH = 64


def solver_base():
    if "benchmark_utils.base_solver" not in sys.modules:
        root, module = types.ModuleType("benchmark_utils"), types.ModuleType("benchmark_utils.base_solver")

        class CompetSolver:
            pass

        module.CompetSolver = CompetSolver
        root.base_solver = module
        sys.modules["benchmark_utils"], sys.modules["benchmark_utils.base_solver"] = root, module


def import_submission(folder, tag):
    solver_base()
    spec = importlib.util.spec_from_file_location(f"{tag}_submission", Path(folder) / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(folder, device, data, tag="main", **changes):
    ch_names = list(changes.get("ch_names", data.channels))
    meta = {"submission_dir": Path(folder), "device": device, "ch_names": ch_names, "n_chans": len(ch_names),
            "chs_info": [{"ch_name": c} for c in ch_names], "n_times": changes.get("n_times", data.n_times),
            "sfreq": changes.get("sfreq", data.sfreq), "n_outputs": changes.get("n_outputs", data.d_out)}
    return import_submission(folder, tag).Solver().load_model(meta)


def refuses(fn):
    try:
        fn()
    except RuntimeError:
        return True
    return False


def contract_checks(folder, data, device, seed=0):
    model = load(folder, device, data)
    gen = torch.Generator().manual_seed(seed)
    rows = np.sort(np.random.default_rng(seed + 8).permutation(len(data.test_subject))[:256])
    x = torch.from_numpy(data.test_x[rows].astype(np.float32)).to(device)
    C, T, D = len(data.channels), data.n_times, data.d_out
    first, second = model.predict(x), model.predict(x)
    order = torch.randperm(len(x), generator=gen).to(device)
    montage = torch.randperm(C, generator=gen)
    shuffled = load(folder, device, data, "montage", ch_names=[data.channels[i] for i in montage.tolist()]).predict(
        x[:, montage.to(device)])
    longer = load(folder, device, data, "window", n_times=T + 1).predict(torch.cat([x, x[:, :, -1:]], dim=2))
    return {"models": len(model.members),
            "self-test cosine": import_submission(folder, "cosine").self_test(model, folder, 0.0),
            "same output on a second call (max difference)": (first - second).abs().max().item(),
            "same output with a shuffled montage (max difference)": (shuffled - first).abs().max().item(),
            "same output with a longer window (max difference)": (longer - first).abs().max().item(),
            "same output in another trial order (max difference)": (model.predict(x[order]) - first[order]).abs().max().item(),
            "output shape for batches of 1 to 200 trials": all(model.predict(x[:b]).shape == (b, D)
                                                               for b in (1, 2, 16, 64, 200) if b <= len(x)),
            "refuses an unknown montage": refuses(lambda: load(folder, device, data, "bad", ch_names=["zz"] + data.channels[1:])),
            "refuses another sampling rate": refuses(lambda: load(folder, device, data, "bad", sfreq=data.sfreq + 8)),
            "refuses a shorter window": refuses(lambda: load(folder, device, data, "bad", n_times=T - 1)),
            "refuses another embedding size": refuses(lambda: load(folder, device, data, "bad", n_outputs=D + 1))}


def public_test(folder, data, devices, seed=0):
    order = torch.randperm(len(data.test_subject), generator=torch.Generator().manual_seed(seed)).numpy()
    batches = [np.sort(order[a:a + BATCH]) for a in range(0, len(order), BATCH)]
    pred = np.empty((len(order), data.d_out), np.float32)
    hb = Progress(f"public test trials in shuffled batches of {BATCH}", len(order))

    def lane(share, device):
        model = load(folder, device, data, f"lane{share}")
        seconds = 0.0
        for rows in batches[share::len(devices)]:
            x = torch.from_numpy(data.test_x[rows].astype(np.float32)).to(device)
            t0 = time.time()
            out = model.predict(x).cpu().numpy()
            seconds += time.time() - t0
            pred[rows] = out
            hb(len(rows))
        return seconds

    seconds = sum(run_on_devices(list(range(len(devices))), lane, devices))
    scores = retrieval_scores(pred, data.test_subject, data.test_image, data.targets, devices[0])
    return {"minutes for 100,000 trials on one device": seconds / len(order) * 1e5 / 60, **scores}


def check(folder, data, devices):
    row = contract_checks(folder, data, devices[0])
    row.update(public_test(folder, data, devices))
    print(pd.Series(row).to_string(), flush=True)
    failed = [k for k, v in row.items() if (k.startswith("same output") and not v <= TOLERANCE) or
              ((k.startswith("refuses") or k.startswith("output shape")) and v is not True)]
    assert not failed, f"checks failed: {failed}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = EEGData(args.data, pool=False)
    source = Path(args.submission)
    if source.is_dir():
        check(source, data, available_devices())
        return
    folder = Path(tempfile.mkdtemp())
    zipfile.ZipFile(source).extractall(folder)
    check(folder, data, available_devices())
    shutil.rmtree(folder)


if __name__ == "__main__":
    main()
