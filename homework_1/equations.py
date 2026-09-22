"""Closed-form analytical model of the HW1 network.

All functions accept Python numbers or NumPy arrays for `image_size` and `batch`
(broadcasting works, so a whole (S, B) surface is one call).

Derivation summary (see hw1_handwritten.pdf for the full version)
-----------------------------------------------------------------
Resolutions (S multiple of 16): conv1 out S/2, pool out S/4, conv5 out S/4,
conv3s2 out S/8, conv1x1 out S/8, conv3s2' out S/16, conv1x1' out S/16.

FLOPs (1 MAC = 2 FLOPs; only conv + linear MACs counted, ReLU/pool/GAP/bias
adds excluded):
    per conv layer  F_l = 2 * B * C_out * H_out * W_out * (C_in * k * k)
    FLOPs(S, B) = B * (17712 * S^2 + 313344)

Peak memory (bytes, torch.cuda.max_memory_allocated during one inference-mode
forward; in-place ReLU, no autograd graph, caller holds the input):
    the peak concurrent pair is conv1's (input, output):
        B*3*S^2*4 + B*32*(S/2)^2*4 = 44*B*S^2 bytes
    Memory(S, B) = 4*1040324 + 44*B*S^2          (lower-bound model)

Bytes moved (DRAM traffic estimate: every activation read+written once,
in-place ReLU still reads and writes, weights read once per forward):
    Bytes(S, B) = 364*B*S^2 + 8592*B + 4161296

Latency (per-operator roofline, theta = {F_eff, BW_eff, c}):
    T(S, B) = sum_i max(F_i / F_eff, Q_i / BW_eff) + c
    where i runs over the 17 launched ops (convs, ReLUs, pool, GAP, linears)
    and c absorbs per-kernel launch overhead (kernel count is ~constant
    across the grid, so a single constant suffices).

Energy (theta_energy = {e_flop, e_byte, E0}):
    E(S, B) = e_flop * FLOPs + e_byte * Bytes + E0
"""

import numpy as np

# ---------------------------------------------------------------------------
# Layer table.
# For op i:   F_i = f_s2 * B * S^2 + f_b * B          (FLOPs, 1 MAC = 2 FLOPs)
#             Q_i = q_s2 * B * S^2 + q_b * B + q_c    (bytes moved, FP32)
# f_s2 etc. derived by hand in hw1_handwritten.pdf.
# ---------------------------------------------------------------------------

# (name,              f_s2,  f_b,     q_s2, q_b,   q_c)
_OP_SPECS = [
    ("conv1_7x7_s2",   2352,  0,       44,   0,     18816),
    ("relu1",          0,     0,       64,   0,     0),
    ("maxpool1",       0,     0,       40,   0,     0),
    ("conv2_5x5",      6400,  0,       24,   0,     204800),
    ("relu2",          0,     0,       32,   0,     0),
    ("conv3_3x3_s2",   2304,  0,       24,   0,     294912),
    ("relu3",          0,     0,       16,   0,     0),
    ("conv4_1x1",      1024,  0,       24,   0,     131072),
    ("relu4",          0,     0,       32,   0,     0),
    ("conv5_3x3_s2",   4608,  0,       20,   0,     2359296),
    ("relu5",          0,     0,       8,    0,     0),
    ("conv6_1x1",      1024,  0,       12,   0,     524288),
    ("relu6",          0,     0,       16,   0,     0),
    ("gap",            0,     0,       8,    2048,  0),
    ("fc1",            0,     262144,  0,    3072,  525312),
    ("relu7",          0,     0,       0,    2048,  0),
    ("fc2",            0,     51200,   0,    1424,  102800),
]

PARAM_COUNT = 1_040_324          # weights + linear biases
PARAM_BYTES = 4 * PARAM_COUNT    # 4_161_296
NUM_KERNELS = len(_OP_SPECS)     # ~17 kernels per forward (cudnn may split some)

# Sanity: the table must reproduce the hand-derived closed forms.
assert sum(s[1] for s in _OP_SPECS) == 17712
assert sum(s[2] for s in _OP_SPECS) == 313344
assert sum(s[3] for s in _OP_SPECS) == 364
assert sum(s[4] for s in _OP_SPECS) == 8592
assert sum(s[5] for s in _OP_SPECS) == PARAM_BYTES


def _prep(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    S, B = np.broadcast_arrays(S, B)
    return S, B


def flops(image_size, batch):
    """FLOPs of one forward pass (1 MAC = 2 FLOPs; conv + linear only)."""
    S, B = _prep(image_size, batch)
    return B * (17712.0 * S**2 + 313344.0)


def memory(image_size, batch):
    """Predicted peak of torch.cuda.max_memory_allocated (bytes), one forward.

    weights + (input + conv1 output) = 4*1040324 + 44*B*S^2.
    Lower-bound model: cuDNN workspace and allocator rounding not included.
    """
    S, B = _prep(image_size, batch)
    return PARAM_BYTES + 44.0 * B * S**2


def bytes_moved(image_size, batch):
    """Estimated DRAM bytes moved by one forward pass (FP32).

    Activations (incl. in-place ReLU read+write) + weights read once:
    364*B*S^2 + 8592*B + 4161296.
    """
    S, B = _prep(image_size, batch)
    return 364.0 * B * S**2 + 8592.0 * B + float(PARAM_BYTES)


def per_op_work(image_size, batch):
    """Per-op FLOPs and bytes. Returns (names, F, Q) with F, Q broadcast arrays."""
    S, B = _prep(image_size, batch)
    names, Fs, Qs = [], [], []
    for (name, f_s2, f_b, q_s2, q_b, q_c) in _OP_SPECS:
        names.append(name)
        Fs.append(f_s2 * B * S**2 + f_b * B)
        Qs.append(q_s2 * B * S**2 + q_b * B + q_c)
    return names, np.stack(Fs), np.stack(Qs)


def latency(image_size, batch, theta):
    """Predicted wall-clock latency (seconds) of one forward pass.

    theta = {"F_eff": achievable FP32 throughput [FLOP/s],
             "BW_eff": achievable DRAM bandwidth [bytes/s],
             "c": fixed launch/launch-overhead floor [s]}

    Per-operator roofline:  T = sum_i max(F_i/F_eff, Q_i/BW_eff) + c.
    """
    S, B = _prep(image_size, batch)
    F_eff = float(np.asarray(theta["F_eff"], dtype=np.float64))
    BW_eff = float(np.asarray(theta["BW_eff"], dtype=np.float64))
    c = float(np.asarray(theta["c"], dtype=np.float64))
    _, F, Q = per_op_work(S, B)
    t = np.maximum(F / F_eff, Q / BW_eff)
    return t.sum(axis=0) + c


def energy(image_size, batch, theta_energy):
    """Predicted energy (joules) of one forward pass, whole GPU.

    theta_energy = {"e_flop": J per FLOP, "e_byte": J per byte moved, "E0": J}
    E = e_flop * FLOPs + e_byte * Bytes + E0.
    """
    S, B = _prep(image_size, batch)
    e_flop = float(np.asarray(theta_energy["e_flop"], dtype=np.float64))
    e_byte = float(np.asarray(theta_energy["e_byte"], dtype=np.float64))
    E0 = float(np.asarray(theta_energy["E0"], dtype=np.float64))
    return e_flop * flops(S, B) + e_byte * bytes_moved(S, B) + E0


def regime(image_size, batch, theta):
    """Classify each (S, B) point: 0 = launch-bound, 1 = memory-bound,
    2 = compute-bound (which per-op roofline term dominates the total)."""
    S, B = _prep(image_size, batch)
    F_eff = float(theta["F_eff"]); BW_eff = float(theta["BW_eff"]); c = float(theta["c"])
    _, F, Q = per_op_work(S, B)
    t_mem = (Q / BW_eff).sum(axis=0)
    t_cmp = (F / F_eff).sum(axis=0)
    out = np.where(t_cmp > t_mem, 2, 1)                     # compute vs memory
    out = np.where(t_cmp + t_mem <= 2.0 * c, 0, out)        # overhead floor
    return out


if __name__ == "__main__":
    # quick self-test against torch's own FLOP counter
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from models import build_model

    model = build_model()
    for (S, B) in [(32, 1), (64, 3), (224, 7), (512, 2)]:
        x = torch.randn(B, 3, S, S, device=next(model.parameters()).device)
        m = FlopCounterMode(display=False)
        with m, torch.inference_mode():
            model(x)
        pred = float(flops(S, B))
        meas = float(m.get_total_flops())
        print(f"S={S:4d} B={B:3d}: predicted {pred:.6e}  measured {meas:.6e}  "
              f"rel.err={(pred-meas)/meas:+.2e}")
        assert abs(pred - meas) / meas < 1e-9
    print("FLOPs equation matches torch FlopCounterMode exactly.")
    # broadcasting demo
    Sg, Bg = np.meshgrid(np.array([32., 128., 512.]), np.array([1., 16., 256.]), indexing="ij")
    print("broadcast shapes:", flops(Sg, Bg).shape, latency(Sg, Bg,
          {"F_eff": 3.5e12, "BW_eff": 2.2e11, "c": 8e-5}).shape)
