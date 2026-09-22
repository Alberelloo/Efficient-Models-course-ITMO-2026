"""Measure the HW1 network on a real GPU over the 132-configuration grid.

Grid: base sizes {32,64,128,224,256,384,512} x base batches
{1,2,4,8,16,32,64,128,256}, plus 4 random multiples of 16 in [32,512] and
3 random non-power-of-2 batches in [1,256] -> full 11 x 12 = 132 configs.

For every config we record (results/measurements.csv):
    S, B, status (ok / oom), latency_s (median, CUDA events),
    memory_bytes (torch.cuda.max_memory_allocated over one forward),
    energy_j (NVML power integrated over K passes / K, whole GPU),
    flops_measured (torch FlopCounterMode), is_validation, gpu info.

Kernel names per config are written to results/kernels.csv via the PyTorch
profiler + record_function layer annotations (separate from timed runs).

Usage (Colab / Kaggle / any CUDA machine):
    python measure.py                    # full grid -> results/
    python measure.py --quick            # tiny smoke grid
    python measure.py --cpu-smoke        # no-GPU smoke test (latency only)

OOMs are caught, recorded, and the run continues (gc + empty_cache).
"""

import argparse
import csv
import gc
import json
import os
import random
import threading
import time

import numpy as np
import torch
import torch.cuda

from models import build_model
from equations import flops as flops_pred, memory as memory_pred

BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_RAND_SIZES, N_RAND_BATCHES = 4, 3


def make_grid(seed: int):
    rng = random.Random(seed)
    rand_sizes = rng.sample([s for s in range(32, 513, 16) if s not in BASE_SIZES],
                            N_RAND_SIZES)
    rand_batches = rng.sample([b for b in range(1, 257)
                               if b not in BASE_BATCHES], N_RAND_BATCHES)
    sizes = sorted(BASE_SIZES + rand_sizes)
    batches = sorted(BASE_BATCHES + rand_batches)
    rand_sizes, rand_batches = sorted(rand_sizes), sorted(rand_batches)
    configs = [(s, b) for s in sizes for b in batches]
    return configs, rand_sizes, rand_batches


# --------------------------------------------------------------------------
# NVML power sampling
# --------------------------------------------------------------------------

class PowerSampler:
    """Samples GPU power (mW) with timestamps until stop() is called."""

    def __init__(self, nvml_handle):
        self.handle = nvml_handle
        self.samples = []          # (t_monotonic_s, power_W)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import pynvml
        while not self._stop.is_set():
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W
            except Exception:
                return
            self.samples.append((time.monotonic(), p))
            time.sleep(0.002)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def energy_joules(self):
        """Trapezoid integral of power over the sampled window (J)."""
        s = self.samples
        if len(s) < 2:
            return float("nan")
        t = np.array([x[0] for x in s]); p = np.array([x[1] for x in s])
        return float(np.trapezoid(p, t))


def nvml_handle():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def measure_idle_power(handle, seconds: float = 0.5) -> float:
    with PowerSampler(handle) as ps:
        time.sleep(seconds)
    ws = [p for _, p in ps.samples]
    return float(np.median(ws)) if ws else float("nan")


# --------------------------------------------------------------------------
# Per-config measurement
# --------------------------------------------------------------------------

def time_forward(model, x, iters: int, warmup: int = 5):
    """Median latency of single forwards (CUDA events), seconds."""
    for _ in range(warmup):
        with torch.inference_mode():
            model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        with torch.inference_mode():
            model(x)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e-3)
    return float(np.median(times))


def choose_iters(est_s: float) -> int:
    """Enough repetitions to time reliably, capped so the run finishes."""
    return int(np.clip(round(0.4 / max(est_s, 1e-6)), 10, 100))


def profile_kernels(model, x, device) -> list:
    """[(layer, kernel_name, count, mean_us)] for one forward, via chrome trace."""
    import tempfile
    from torch.profiler import profile, ProfilerActivity

    # Explicit record_function scopes give us layer names in the trace.
    feats = model.features
    names = ["conv1_7x7_s2", "relu1", "maxpool1", "conv2_5x5", "relu2",
             "conv3_3x3_s2", "relu3", "conv4_1x1", "relu4", "conv5_3x3_s2",
             "relu5", "conv6_1x1", "relu6", "gap", "fc1", "relu7", "fc2"]

    def annotated_forward(x):
        from torch.profiler import record_function
        out = x
        for i, m in enumerate(feats):
            with record_function(names[i]):
                out = m(out)
        with record_function("gap"):
            out = model.pool(out).flatten(1)
        with record_function("fc1"):
            out = model.classifier[0](out)
        with record_function("relu7"):
            out = model.classifier[1](out)
        with record_function("fc2"):
            out = model.classifier[2](out)
        return out

    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.inference_mode():
                annotated_forward(x)
            torch.cuda.synchronize()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        prof.export_chrome_trace(path)
        with open(path) as f:
            trace = json.load(f)
        os.unlink(path)

        anns = []   # (name, ts, dur) for user annotations
        kernels = []  # (ts, dur, name)
        for ev in trace.get("traceEvents", []):
            cat = ev.get("cat", "")
            if cat in ("user_annotation", "gpu_user_annotation") and "dur" in ev:
                anns.append((ev["name"], ev["ts"], ev["ts"] + ev["dur"], cat))
            elif cat in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in ev:
                kernels.append((ev["ts"], ev["ts"] + ev["dur"], ev["name"]))
        # prefer device-side annotations (kernels correlate with them directly)
        if any(a[3] == "gpu_user_annotation" for a in anns):
            anns = [a for a in anns if a[3] == "gpu_user_annotation"]
        anns = [(n, t0, t1) for (n, t0, t1, _) in anns]
        rows = {}
        for (a_name, a0, a1) in anns:
            for (k0, k1, k_name) in kernels:
                if k0 >= a0 - 0.5 and k1 <= a1 + 0.5:
                    key = (a_name, k_name)
                    n, tot = rows.get(key, (0, 0.0))
                    rows[key] = (n + 1, tot + (k1 - k0))
        return [(ln, kn, n, tot / n) for (ln, kn), (n, tot) in sorted(rows.items())]
    except Exception as e:
        print(f"    [warn] kernel profiling failed: {e}")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny smoke grid")
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="run on CPU without memory/energy/kernel measurement")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)

    device = "cpu" if args.cpu_smoke else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available (use --cpu-smoke for a CPU test)")

    configs, rand_sizes, rand_batches = make_grid(args.seed)
    if args.quick or args.cpu_smoke:
        configs = [(s, b) for (s, b) in configs
                   if s in (32, 64, 128) and b in (1, 8, 32)][:12]

    model = build_model().to(device)
    model.eval()

    handle = None
    idle_power = float("nan")
    gpu_name, gpu_total = "cpu", 0
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        handle = nvml_handle()
        if handle is not None:
            idle_power = measure_idle_power(handle)
            print(f"GPU: {gpu_name}, {gpu_total/2**30:.1f} GiB, "
                  f"idle power ~{idle_power:.1f} W")

    meta = {
        "gpu_name": gpu_name,
        "gpu_total_bytes": int(gpu_total),
        "idle_power_w": idle_power,
        "torch": torch.__version__,
        "seed": args.seed,
        "rand_sizes": rand_sizes,
        "rand_batches": rand_batches,
        "cudnn_benchmark": False,
        "allow_tf32": False,
    }
    with open(os.path.join(args.output, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    mem_rows, kern_rows = [], []
    t_start = time.time()

    for idx, (S, B) in enumerate(configs):
        tag = f"[{idx+1:3d}/{len(configs)}] S={S:4d} B={B:4d}"
        rec = dict(S=S, B=B,
                   is_validation=int(S not in BASE_SIZES or B not in BASE_BATCHES),
                   status="ok", latency_s="", memory_bytes="", energy_j="",
                   flops_measured="", flops_predicted=flops_pred(S, B),
                   memory_predicted=memory_pred(S, B))

        x = None
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            x = torch.randn(B, 3, S, S, device=device)

            # ---- FLOPs via torch's counter (independent of our formula) ----
            from torch.utils.flop_counter import FlopCounterMode
            m = FlopCounterMode(display=False)
            with m, torch.inference_mode():
                model(x)
            rec["flops_measured"] = m.get_total_flops()

            # ---- peak memory over exactly one forward ----
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    model(x)
                torch.cuda.synchronize()
                rec["memory_bytes"] = torch.cuda.max_memory_allocated()

            # ---- latency (median over repeated single forwards) ----
            if device == "cuda":
                est = time_forward(model, x, iters=5, warmup=2)
                iters = choose_iters(est)
                rec["latency_s"] = time_forward(model, x, iters=iters)
            else:
                with torch.inference_mode():
                    model(x)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model(x)
                rec["latency_s"] = time.perf_counter() - t0

            # ---- energy (NVML, whole GPU, integrated over K passes) ----
            if device == "cuda" and handle is not None:
                K = int(np.clip(round(0.5 / max(rec["latency_s"], 1e-6)), 5, 200))
                torch.cuda.synchronize()
                time.sleep(0.05)
                with PowerSampler(handle) as ps:
                    for _ in range(K):
                        with torch.inference_mode():
                            model(x)
                    torch.cuda.synchronize()
                rec["energy_j"] = ps.energy_joules() / K

            # ---- kernel names (profiled separately; not timed) ----
            if device == "cuda" and not args.quick:
                for (layer, kname, cnt, mean_us) in profile_kernels(model, x, device):
                    kern_rows.append(dict(S=S, B=B, layer=layer, kernel=kname,
                                          count=cnt, mean_us=mean_us))

            print(f"{tag} ok  lat={rec['latency_s']*1e3:9.3f} ms  "
                  f"mem={float(rec['memory_bytes'] or 0)/2**20:8.1f} MiB  "
                  f"E={float(rec['energy_j'] or 'nan'):8.4f} J")

        except torch.cuda.OutOfMemoryError as e:
            rec["status"] = "oom"
            print(f"{tag} OOM (predicted {rec['memory_predicted']/2**30:.2f} GiB)")
        except RuntimeError as e:
            rec["status"] = f"error:{str(e)[:60]}"
            print(f"{tag} ERROR {e}")
        finally:
            del x
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        mem_rows.append(rec)

    # ---------------- write outputs ----------------
    fields = ["S", "B", "status", "latency_s", "memory_bytes", "energy_j",
              "flops_measured", "flops_predicted", "memory_predicted",
              "is_validation"]
    with open(os.path.join(args.output, "measurements.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(mem_rows)

    if kern_rows:
        with open(os.path.join(args.output, "kernels.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["S", "B", "layer", "kernel",
                                              "count", "mean_us"])
            w.writeheader()
            w.writerows(kern_rows)

    print(f"\nDone: {len(configs)} configs in {(time.time()-t_start)/60:.1f} min "
          f"-> {args.output}/measurements.csv, kernels.csv, run_meta.json")
    n_ok = sum(r["status"] == "ok" for r in mem_rows)
    n_oom = sum(r["status"] == "oom" for r in mem_rows)
    print(f"ok={n_ok}  oom={n_oom}")


if __name__ == "__main__":
    main()
