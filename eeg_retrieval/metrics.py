import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def tie_ranks(Q, pos, G, chunk=8192):
    out = torch.empty(len(Q), device=Q.device)
    for a in range(0, len(Q), chunk):
        S = F.normalize(Q[a:a + chunk].to(G.dtype), dim=1) @ G.T
        t = S.gather(1, pos[a:a + chunk, None])
        out[a:a + chunk] = ((S > t).sum(1) + (S >= t).sum(1) - 1).float() / 2
    return out


@torch.no_grad()
def retrieval_scores(pred, subject, image, targets, device):
    gallery = np.unique(image)
    G = F.normalize(torch.from_numpy(targets[gallery]).double().to(device), dim=1)
    pos = torch.from_numpy(np.searchsorted(gallery, image)).to(device)
    people = np.unique(subject)
    key = torch.from_numpy(np.searchsorted(people, subject)).to(device) * len(gallery) + pos
    uniq, inv = torch.unique(key, return_inverse=True)
    sums = torch.zeros(len(uniq), pred.shape[1], dtype=torch.float64, device=device)
    single = torch.empty(len(pred), device=device)
    for a in range(0, len(pred), 32768):
        p = torch.as_tensor(pred[a:a + 32768]).double().to(device)
        sums.index_add_(0, inv[a:a + 32768], p)
        single[a:a + 32768] = tie_ranks(p, pos[a:a + 32768], G)
    counts = torch.bincount(inv, minlength=len(uniq)).double()
    averaged = tie_ranks(sums / counts[:, None], uniq % len(gallery), G)
    return {"top5_trial": (single < 5).double().mean().item(), "top1_trial": (single < 1).double().mean().item(),
            "top5_averaged": (averaged < 5).double().mean().item(), "top1_averaged": (averaged < 1).double().mean().item(),
            "n_candidates": len(gallery)}


def presentation_queries(subject, image, held, gallery, seed, few=2, max_groups=40):
    cell = np.searchsorted(held, subject) * len(gallery) + np.searchsorted(gallery, image)
    order = np.lexsort((np.random.default_rng(seed).random(len(cell)), cell))
    starts = np.r_[0, np.flatnonzero(np.diff(cell[order])) + 1]
    rank = np.empty(len(cell), np.int64)
    rank[order] = np.arange(len(cell)) - np.repeat(starts, np.diff(np.r_[starts, len(cell)]))
    n_cell = np.bincount(cell, minlength=len(held) * len(gallery))[cell]
    width = int(n_cell.max()) + 1
    ids = {"1": cell * width + rank,
           str(few): np.where(rank // few < np.minimum(max_groups, n_cell // few), cell * width + rank // few, -1),
           "all": cell * width}
    out = {}
    for name, qid in ids.items():
        rows = np.flatnonzero(qid >= 0)
        uniq, inv = np.unique(qid[rows], return_inverse=True)
        qcell = uniq // width
        out[name] = {"rows": torch.from_numpy(rows), "inv": torch.from_numpy(inv), "n": len(uniq),
                     "person": torch.from_numpy(qcell // len(gallery)), "image": torch.from_numpy(qcell % len(gallery))}
    return out


def per_person(ranks, person, n_people, cuts):
    count = torch.bincount(person, minlength=n_people).float()
    return {t: (torch.zeros(n_people, device=ranks.device).index_add_(0, person, (ranks < t).float()) / count).cpu().numpy()
            for t in cuts}


@torch.no_grad()
def score_queries(unit, queries, n_people, G_test, G_all, pos_all, device):
    out = {}
    for name, q in queries.items():
        sums = torch.zeros(q["n"], unit.shape[1], device=device).index_add_(0, q["inv"].to(device), unit[q["rows"].to(device)].float())
        person, image = q["person"].to(device), q["image"].to(device)
        small = per_person(tie_ranks(sums, image, G_test), person, n_people, (1, 5))
        large = per_person(tie_ranks(sums, pos_all[image], G_all), person, n_people, (5, 50))
        for t, v in small.items():
            out[("test images", f"top-{t}", name)] = v
        for t, v in large.items():
            out[("all images", f"top-{t}", name)] = v
    return out
