import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch


def available_devices():
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]


def release(device):
    if device.startswith("cuda"):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def run_on_devices(jobs, fn, devices):
    free = queue.Queue()
    for d in devices:
        free.put(d)

    def worker(job):
        device = free.get()
        try:
            return fn(job, device)
        finally:
            release(device)
            free.put(device)

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        return list(ex.map(worker, jobs))


class Progress:
    def __init__(self, label, total, every=60):
        self.label, self.total, self.done, self.every = label, max(1, total), 0, every
        self.t0 = self.last = time.time()
        self.lock = threading.Lock()

    def __call__(self, n=1):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last >= self.every or self.done >= self.total:
                rate = self.done / max(now - self.t0, 1e-9)
                print(f"  [{self.label}] {self.done:,}/{self.total:,} | {rate:,.1f}/s | {(now - self.t0) / 60:.1f} min"
                      f" | {(self.total - self.done) / max(rate, 1e-9) / 60:.1f} min left", flush=True)
                self.last = now
