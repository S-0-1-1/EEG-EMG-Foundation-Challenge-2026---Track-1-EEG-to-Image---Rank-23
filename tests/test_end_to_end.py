import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "O1", "O2"]
WIDE = ["F3", "O2", "Cz", "F4", "Oz", "C4", "Pz", "P3", "Fz", "O1", "C3", "P4"]
TRAINING = {"epochs": 1, "samples_per_epoch": 96, "batch_size": 16, "lr_head": 0.001, "lr_encoder": 0.0003, "weight_decay": 0.01,
            "warmup_steps": 2, "grad_clip": 1.0, "temperature": 0.07, "cosine_weight": 0.5, "transport_p": 0.5, "shift_p": 0.5,
            "shift_max": 2, "blend_p": 0.5, "blend_max": 0.5, "covariance_trials": 40, "monitor_people": 2, "epoch_hours": 0.01}


def simulate(root, channels, n_times, seed, people=5, train_images=24, test_images=6, width=32):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    n_images = train_images + test_images
    targets = rng.standard_normal((n_images, width)).astype(np.float32)
    templates = rng.standard_normal((n_images, len(channels), n_times)).astype(np.float32)
    gains = 1 + 0.3 * rng.standard_normal((people, len(channels), 1)).astype(np.float32)

    def trials(images, repeats):
        subject = np.repeat(np.arange(1, people + 1), len(images) * repeats)
        image = np.tile(np.repeat(images, repeats), people)
        x = gains[subject - 1] * templates[image] + rng.standard_normal((len(image), len(channels), n_times)).astype(np.float32)
        return x.astype(np.float16), subject.astype(np.int16), image.astype(np.int32)

    splits = {"train": trials(np.arange(train_images), 3), "val": trials(np.arange(train_images), 1),
              "test": trials(np.arange(train_images, n_images), 4)}
    for name, (x, subject, image) in splits.items():
        np.save(root / f"{name}_X.npy", x)
        np.save(root / f"{name}_subject.npy", subject)
        np.save(root / f"{name}_image.npy", image)
    np.save(root / "targets_dinov2_giant.npy", targets)
    (root / "meta.json").write_text(json.dumps({"ch_names": channels, "sfreq": 120.0}))


def model(name, start, cosine, transport, blend, folds):
    return {"name": name, "start": start, "cosine": cosine, "transport": transport, "blend": blend, "seed": 0, "folds": folds}


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=REPO)


def test_end_to_end():
    work = Path(tempfile.mkdtemp())
    simulate(work / "main", MAIN, 120, 0)
    simulate(work / "wide", WIDE, 132, 1)
    first = work / "first.json"
    second = work / "second.json"
    first.write_text(json.dumps({"training": TRAINING, "models": [model("first", "random", "uncentered", False, False, False)]}))
    second.write_text(json.dumps({"training": TRAINING, "models": [model("full", "first", "centered", True, True, True),
                                                                   model("contrast", "random", "none", False, False, False)]}))
    out = str(work / "checkpoints")
    run("eeg_retrieval.train", "--config", str(first), "--data", str(work / "wide"), "--channels-from", str(work / "main"),
        "--out", out)
    run("eeg_retrieval.train", "--config", str(second), "--data", str(work / "main"), "--out", out, "--mode", "both")
    run("eeg_retrieval.evaluate", "--data", str(work / "main"), "--predictions", str(work / "checkpoints" / "folds"),
        "--models", "full", "--ensemble", "full", "full", "--out", str(work / "scores.csv"))
    run("eeg_retrieval.package", "--config", str(second), "--checkpoints", out, "--data", str(work / "main"),
        "--out", str(work / "submission"))
    run("eeg_retrieval.check", "--submission", str(work / "submission.zip"), "--data", str(work / "main"))
    assert (work / "submission.zip").exists() and (work / "scores.csv").exists()
    print(f"all steps passed | files in {work}")


if __name__ == "__main__":
    test_end_to_end()
