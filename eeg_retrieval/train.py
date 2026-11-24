import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import N_FOLDS, EEGData, layout
from .progress import available_devices
from .training import PUBLISHED, Session, check_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("all", "folds", "both"), default="all")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--channels-from")
    ap.add_argument("--hours", type=float)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    specs = [s for s in config["models"] if not args.only or s["name"] in args.only]
    check_specs(specs)
    assert specs, "no model selected"
    out = Path(args.out)
    for s in specs:
        ready = s["start"] in PUBLISHED or all((out / f"{s['start']}.{e}").exists() for e in ("pt", "json"))
        assert ready, f"{s['name']} starts from {s['start']}: train that model first, into {out}"
    like = layout(args.channels_from) if args.channels_from else None
    torch.backends.cudnn.benchmark = True
    data = EEGData(args.data) if like is None else EEGData(args.data, channels=like["channels"], n_times=like["n_times"])
    assert like is None or abs(data.sfreq - like["sfreq"]) < 1e-6, f"sampling rate {data.sfreq} instead of {like['sfreq']}"
    session = Session(data, config["training"], out, hours=args.hours)
    jobs = []
    if args.mode in ("folds", "both"):
        jobs += [(s["name"], k) for s in specs if s.get("folds", False) for k in range(N_FOLDS)]
    if args.mode in ("all", "both"):
        jobs += [(s["name"], "all") for s in specs]
    left = session.run({s["name"]: s for s in specs}, jobs, available_devices())
    print(f"still to run: {left}")
    summary = out / "summary.csv"
    if summary.exists():
        S = pd.read_csv(summary)
        print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        folds = S[S["fold"].astype(str) != "all"]
        if len(folds):
            print(folds.groupby("model")[["top5_averaged", "top5_trial"]].agg(["mean", "count"]).to_string(
                float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
