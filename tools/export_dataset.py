import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

STUDIES = {"alljoined-1.6m": "xu2025alljoined", "things-eeg2": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=sorted(STUDIES), required=True)
    ap.add_argument("--competition-repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    work, out = Path(args.data_dir).resolve(), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(user="local", entity_name="local", project_name="neuralbench", data_dir=str(work),
                    cache_dir=str(work / "cache"), save_dir=str(work / "save"), n_cpus=args.workers, cluster="auto")
    (work / "neuralbench.json").write_text(json.dumps({k.upper(): v for k, v in settings.items()}, indent=2))
    os.environ["NEURALBENCH_CONFIG"] = str(work / "neuralbench.json")
    os.environ["NEURALBENCH_EXTRACT_CPUS"] = str(args.workers)

    sys.path.insert(0, args.competition_repo)
    from benchmark_utils.nb_task import download_study, load_task

    overlay = STUDIES[args.study]
    if args.download:
        download_study("eeg", "image", work, dataset=overlay)
    t0 = time.time()
    loaders, meta = load_task("eeg", "image", data_dir=str(work), dataset=overlay, batch_size=512, num_workers=0)
    print(f"prepared in {(time.time() - t0) / 60:.1f} min | {meta['n_chans']} channels x {meta['n_times']} samples at "
          f"{meta['sfreq']:g} Hz | target {meta['target_shape']}", flush=True)
    (out / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "device"}, indent=2, default=str))

    targets, index = [], {}
    for split in ("train", "val", "test"):
        ds = loaders[split].dataset
        n = len(ds)
        X = np.lib.format.open_memmap(out / f"{split}_X.npy", mode="w+", dtype=np.float16, shape=(n, meta["n_chans"], meta["n_times"]))
        subject, image = np.empty(n, np.int16), np.empty(n, np.int32)
        i = 0
        for step, (xb, yb, info) in enumerate(DataLoader(ds, batch_size=512, shuffle=False, num_workers=args.workers)):
            b = xb.shape[0]
            X[i:i + b] = xb.numpy().astype(np.float16)
            subject[i:i + b] = np.asarray(info["subject_id"])
            for j, y in enumerate(yb.numpy().astype(np.float32)):
                key = y.tobytes()
                if key not in index:
                    index[key] = len(targets)
                    targets.append(y)
                image[i + j] = index[key]
            i += b
            if step % 200 == 199:
                print(f"  {split}: {i:,}/{n:,} trials | {(time.time() - t0) / 60:.1f} min", flush=True)
        X.flush()
        np.save(out / f"{split}_subject.npy", subject)
        np.save(out / f"{split}_image.npy", image)
        print(f"{split}: {n:,} trials | {len(np.unique(subject))} people | {len(np.unique(image)):,} images | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    np.save(out / "targets_dinov2_giant.npy", np.stack(targets))
    print(f"images overall: {len(targets):,}")


if __name__ == "__main__":
    main()
