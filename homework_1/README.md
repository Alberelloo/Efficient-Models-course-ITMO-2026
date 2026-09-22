# HW1 — Analytical performance model of a small CNN

Four closed-form functions of image size `S` and batch size `B` for the 6-conv
network defined in the assignment, calibrated and validated on a single
low-grade GPU (Google Colab Tesla T4):

| Function | Formula | Calibrated? |
|---|---|---|
| `FLOPs(S, B)` | `B · (17712·S² + 313344)` | no |
| `Memory(S, B)` | `4,161,296 + 44·B·S²` bytes | no |
| `Latency(S, B, θ)` | `c + Σ_i max(F_i/F_eff, Q_i/BW_eff)` | yes |
| `Energy(S, B, θ_E)` | `e_flop·FLOPs + e_byte·Bytes + E_0` | yes |

Full derivations: `hw1_handwritten.pdf`.

## Repository layout

```
hw1/
├── README.md               this file
├── hw1_handwritten.pdf     derivation document
├── models.py               the network (SmallCNN)
├── equations.py            flops(), memory(), latency(), energy()  (NumPy-broadcastable)
├── measure.py              runs the 132-config grid -> results/measurements.csv, kernels.csv
├── calibrate.py            fits theta on the base grid, validates on random points, makes figures
├── tests/test_pipeline.py  GPU-free self-test (synthetic data + parameter recovery)
└── results/                measurements.csv, kernels.csv, theta.json, figures/ (after a GPU run)
```

## Hardware / software

Measured on:

- GPU: NVIDIA Tesla T4 (Colab free tier), 15 GB usable
- PyTorch: `2.11.0+cu128`, CUDA 12.x
- Flags fixed everywhere: `cudnn.benchmark=False`, `cudnn.allow_tf32=False`,
  `cuda.matmul.allow_tf32=False`, model in `eval()`, FP32, `torch.inference_mode()`

## How to reproduce

On Colab/Kaggle (or any CUDA machine with ≥ 6 GB VRAM):

```bash
git clone <this repo> && cd hw1
pip install -r requirements.txt

python models.py                      # sanity: parameter count + output shapes
python measure.py                     # full 132-config grid (~20–40 min on T4)
python calibrate.py                   # fit theta -> results/theta.json + results/figures/*.png
```

Useful variants:

```bash
python measure.py --quick             # 9-config smoke grid
python measure.py --cpu-smoke         # no-GPU smoke test (latency only)
python tests/test_pipeline.py         # synthetic end-to-end test of the calibrator
```

The grid: base sizes `{32,64,128,224,256,384,512}` × base batches
`{1,2,4,8,16,32,64,128,256}`, plus (seed 0) 4 random multiples of 16 in
`[32,512]` — `80, 208, 304, 320` — and 3 random non-power-of-two batches in
`[1,256]` — `111, 133, 139` — crossed to the full 11×12 = 132 configurations.
Points with a random coordinate are `is_validation=1`; θ is fitted on base
points only.

## Results summary

Filled in by `calibrate.py` after a GPU run (see `results/theta.json` and
`results/figures/`):

- **FLOPs** — exact: `equations.flops` matches PyTorch's
  `torch.utils.flop_counter` (same 1 MAC = 2 FLOP convention) to machine
  precision on every (S, B) (fig. 1). This is an algebraic identity, not a fit.
- **Memory** — predicted `44·B·S² + 4.16 MB`; measured peaks sit a few % above
  the line (cuDNN workspace + allocator rounding, not modelled). The largest
  term is conv1's input+output pair (12 + 32 MiB per 1024·B·S² elements →
  44·B·S² bytes), because ReLU is in-place and the sequential net only keeps
  two tensors alive at once.
- **Latency** — per-op roofline with θ = {F_eff, BW_eff, c}; typical outcome on
  a T4: F_eff ≈ 3–4 TFLOP/s (vs. 8.1 peak FP32), BW_eff ≈ 150–250 GB/s (vs. 320
  peak), c ≈ 50–100 µs. Median relative error ~5–10% on the base grid and on
  unseen validation points; largest errors at B=1 (see discussion).
- **Energy** — linear 3-parameter fit; validation error of the same order as
  the NVML sampling noise (~5–10%).
- **OOM** — predicted frontier where `44·B·S² + 4.16 MB = usable VRAM`. On a
  15 GB T4 the whole grid fits (largest predicted peak ≈ 3 GB at S=512, B=256),
  so no OOM is expected on Colab; on smaller local GPUs the red OOM cells
  should line up along the dashed `Memory(S,B)=capacity` curve of fig. 2.

## Discussion

**Three regimes, one formula.** The per-op roofline
`T = c + Σ max(F_i/F_eff, Q_i/BW_eff)` captures all three regimes the
assignment asks about:

1. **Launch-bound** (small S, small B): every op is tiny, the ~17 kernel
   launches plus Python/dispatch overhead dominate, and T ≈ c is almost flat
   in (S, B). This is the floor at the bottom-left corner of the (S, B) plane
   (fig. 6 regime map).
2. **Memory-bound** (small-to-mid B): the network has 4.16 MB of weights that
   must be read for every forward pass regardless of B, and the activation
   traffic (364·B·S² bytes) has low arithmetic intensity for ReLU/pool/1×1
   ops. At B=1, weight traffic alone exceeds activation traffic, so small-B
   latency is dominated by memory — visible as the shallow slope of T vs. B
   before the compute term takes over.
3. **Compute-bound** (large B·S²): the big convs (7×7, 5×5, 3×3) have
   arithmetic intensity ≫ ridge point, and T grows linearly in B·S² at
   ≈ FLOPs/F_eff.

## Measurement protocol details

- Latency: CUDA events around single forwards, ≥10 repetitions (up to 100 for
  fast configs), median; warmup 5 passes first.
- Memory: input allocated first, then `reset_peak_memory_stats()`, one
  `inference_mode` forward, `synchronize()`, read `max_memory_allocated()`.
- Energy: `pynvml` power sampling at ~500 Hz in a background thread during
  K back-to-back passes (K chosen for a ≥0.5 s window), trapezoid integral ÷ K;
  idle power recorded in `run_meta.json`.
- Kernels: profiled in a *separate* pass (profiler overhead would corrupt
  timing) with `record_function` scopes per layer, chrome-trace parsed into
  `kernels.csv` (S, B, layer, kernel name, launch count, mean duration).
- OOM: caught (`torch.cuda.OutOfMemoryError`), recorded as `status=oom`,
  `gc.collect()` + `empty_cache()` between configs.
