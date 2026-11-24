import numpy as np
import torch


def psd_power(cov, inverse=False):
    w, V = torch.linalg.eigh(cov)
    w = w.clamp_min(1e-10 * w.max())
    return (V * (w.rsqrt() if inverse else w.sqrt())) @ V.T


def transport_maps(x_pool, subject, people, n_trials, seed):
    rng = np.random.default_rng(seed)
    n_chans = x_pool.shape[1]
    covs = []
    for s in people:
        rows = np.flatnonzero(subject == s)
        rows = np.sort(rng.choice(rows, min(n_trials, len(rows)), replace=False))
        x = torch.from_numpy(x_pool[rows].astype(np.float32)).double()
        x = x - x.mean(-1, keepdim=True)
        cov = torch.einsum("nct,ndt->cd", x, x) / (x.shape[0] * x.shape[-1])
        covs.append(cov + 1e-3 * torch.diagonal(cov).mean() * torch.eye(n_chans, dtype=torch.float64))
    maps = torch.zeros(len(people), len(people), n_chans, n_chans, dtype=torch.float64)
    for a in range(len(people)):
        root, inv_root = psd_power(covs[a]), psd_power(covs[a], inverse=True)
        for b in range(len(people)):
            maps[a, b] = inv_root @ psd_power(root @ covs[b] @ root) @ inv_root
    return maps.float()


def person_transport(x, person, maps, gen, p):
    n, device = x.shape[0], x.device
    eye = torch.eye(x.shape[1], device=device)
    on = torch.rand(n, device=device, generator=gen) < p
    other = torch.randint(0, maps.shape[0] - 1, (n,), device=device, generator=gen)
    other = other + (other >= person).long()
    t = torch.rand(n, device=device, generator=gen)[:, None, None]
    A = eye + t * (maps[person, other] - eye)
    return torch.where(on[:, None, None], torch.bmm(A, x), x)


def time_shift(x, gen, p, max_shift):
    n, device = x.shape[0], x.device
    on = torch.rand(n, device=device, generator=gen) < p
    d = torch.randint(1, max_shift + 1, (n,), device=device, generator=gen)
    d = d * (torch.randint(0, 2, (n,), device=device, generator=gen) * 2 - 1) * on.long()
    index = (torch.arange(x.shape[-1], device=device)[None, :] - d[:, None]).clamp(0, x.shape[-1] - 1)
    return x.gather(2, index[:, None, :].expand(-1, x.shape[1], -1))


class SameImageIndex:
    def __init__(self, image, rows):
        self.order = rows[np.argsort(image[rows], kind="stable")]
        sorted_images = image[self.order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_images)) + 1]
        self.first = np.zeros(int(image.max()) + 1, np.int64)
        self.count = np.zeros(int(image.max()) + 1, np.int64)
        self.first[sorted_images[starts]] = starts
        self.count[sorted_images[starts]] = np.diff(np.r_[starts, len(self.order)])

    def draw(self, images, rng):
        return self.order[self.first[images] + (rng.random(len(images)) * self.count[images]).astype(np.int64)]


def blend_same_image(x, other, gen, p, max_weight):
    lam = torch.rand(x.shape[0], device=x.device, generator=gen) * max_weight
    lam = lam * (torch.rand(x.shape[0], device=x.device, generator=gen) < p)
    return x + lam[:, None, None] * (other - x)
