import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import N_FOLDS, EEGData
from .metrics import presentation_queries, score_queries
from .progress import Progress, available_devices, run_on_devices

COUNTS = ("1", "2", "all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ensemble", nargs="+", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()

    data = EEGData(args.data, pool=False)
    folder = Path(args.predictions)
    entries = {m: [m] for m in args.models}
    entries.update({" + ".join(e): e for e in args.ensemble})
    needed = sorted({m for members in entries.values() for m in members})
    absent = [str(folder / f"{m}_fold{k}.npy") for m in needed for k in range(N_FOLDS) if not (folder / f"{m}_fold{k}.npy").exists()]
    assert not absent, f"fold predictions not found: {absent}"
    pos_all = torch.from_numpy(np.searchsorted(data.gallery_all, data.gallery_test))
    unit = data.unit_targets
    hb = Progress("scoring", N_FOLDS * len(entries))

    def fold_job(k, device):
        held, _, te = data.split(k)
        queries = presentation_queries(data.test_subject[te], data.test_image[te], held, data.gallery_test, 10 + k)
        G_test = unit[torch.from_numpy(data.gallery_test)].to(device)
        G_all = unit[torch.from_numpy(data.gallery_all)].to(device)
        preds = {}
        for m in needed:
            p = torch.from_numpy(np.load(folder / f"{m}_fold{k}.npy").astype(np.float32)).to(device)
            assert len(p) == len(te), f"{m} fold {k}: {len(p)} rows for {len(te)} test trials"
            preds[m] = F.normalize(p, dim=1).half()
        rows = []
        for label, members in entries.items():
            fused = preds[members[0]] if len(members) == 1 else F.normalize(sum(preds[m].float() for m in members), dim=1)
            scores = score_queries(fused, queries, len(held), G_test, G_all, pos_all.to(device), device)
            rows += [{"model": label, "gallery": g, "metric": t, "presentations": n, "person": int(s), "value": float(v)}
                     for (g, t, n), values in scores.items() for s, v in zip(held.tolist(), values, strict=True)]
            hb()
        return rows

    results = pd.DataFrame([r for rows in run_on_devices(list(range(N_FOLDS)), fold_job, available_devices()) for r in rows])
    table = results.pivot_table(index="model", columns=["gallery", "metric", "presentations"], values="value", aggfunc="mean")
    table = table.loc[list(entries)]
    blocks = [("test images", "top-5", len(data.gallery_test)), ("test images", "top-1", len(data.gallery_test)),
              ("all images", "top-50", len(data.gallery_all)), ("all images", "top-5", len(data.gallery_all))]
    for gallery, metric, size in blocks:
        chance = min(1.0, int(metric.split("-")[1]) / size)
        print(f"{metric} among {size:,} images (chance {chance:.6f}) | {results['person'].nunique()} held-out people | "
              f"columns: presentations averaged per query")
        print(table[gallery][metric][list(COUNTS)].to_string(float_format=lambda v: f"{v:.5f}"))
    if args.out:
        flat = table.copy()
        flat.columns = [f"{g} {t} {n}" for g, t, n in flat.columns]
        flat.to_csv(args.out)
        results.to_csv(Path(args.out).with_suffix(".people.csv"), index=False)


if __name__ == "__main__":
    main()
