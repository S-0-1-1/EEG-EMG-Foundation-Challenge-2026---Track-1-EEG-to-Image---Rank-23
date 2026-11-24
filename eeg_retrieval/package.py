import argparse
import json
import shutil
import zipfile
from pathlib import Path

import braindecode
import numpy as np
import torch

from . import inference
from .check import check, import_submission
from .data import EEGData
from .progress import available_devices

SOLVER = '''

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "{name}"

    def load_model(self, meta):
        return load_ensemble(meta["submission_dir"], meta)
'''
REFERENCE_TRIALS = 8
SELF_TEST_COSINE = 0.999


def build(folder, names, checkpoints, data, solver_name, seed=0):
    folder, checkpoints = Path(folder), Path(checkpoints)
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "weights").mkdir(parents=True)
    members = []
    for name in names:
        info = json.loads((checkpoints / f"{name}.json").read_text())
        same = [c.lower() for c in info["channels"]] == [c.lower() for c in data.channels]
        assert same and info["n_times"] == data.n_times and info["d_out"] == data.d_out and \
            abs(info["sfreq"] - data.sfreq) < 1e-6, f"{name} was trained on another montage, window, rate or embedding size"
        state = torch.load(checkpoints / f"{name}.pt", map_location="cpu")
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}, folder / "weights" / f"{name}.pt")
        members.append({"name": name, "file": f"weights/{name}.pt", "encoder_config": info["encoder_config"]})
    (folder / "model.json").write_text(json.dumps(
        {"members": members, "channels": data.channels, "n_times": data.n_times, "sfreq": data.sfreq, "d_out": data.d_out,
         "self_test_cosine": SELF_TEST_COSINE, "braindecode": braindecode.__version__, "torch": torch.__version__}, indent=1))
    (folder / "submission.py").write_text(Path(inference.__file__).read_text() + SOLVER.format(name=solver_name))
    module = import_submission(folder, "reference")
    rows = np.sort(np.random.default_rng(seed + 7).permutation(len(data.test_subject))[:REFERENCE_TRIALS])
    x = torch.from_numpy(data.test_x[rows].astype(np.float32))
    with torch.no_grad():
        outs = [module.build_member(m["encoder_config"], folder / m["file"], data.channels, data.n_times, data.sfreq, data.d_out)(x)
                .float() for m in members]
    np.savez(folder / "reference.npz", x=x.numpy(), members=np.stack([o.numpy() for o in outs]),
             ensemble=module.fuse(outs).numpy())
    return folder


def archive(folder, target):
    folder, target = Path(folder), Path(target)
    for cache in folder.rglob("__pycache__"):
        shutil.rmtree(cache)
    made = shutil.make_archive(str(target.with_suffix("")), "zip", folder)
    names = zipfile.ZipFile(made).namelist()
    assert {"submission.py", "model.json", "reference.npz"} <= set(names) and not [n for n in names if "__pycache__" in n]
    return Path(made)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--name", default="eeg-image-retrieval")
    args = ap.parse_args()
    names = [s["name"] for s in json.loads(Path(args.config).read_text())["models"] if not args.only or s["name"] in args.only]
    absent = [n for n in names if not (Path(args.checkpoints) / f"{n}.pt").exists()]
    assert not absent, f"models not trained on all people yet: {absent}"
    data = EEGData(args.data, pool=False)
    folder = build(args.out, names, args.checkpoints, data, args.name)
    check(folder, data, available_devices())
    made = archive(folder, f"{args.out}.zip")
    print(f"{made} | {len(names)} models | {made.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
