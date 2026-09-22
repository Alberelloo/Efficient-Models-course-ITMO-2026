"""Calibrate theta on the base grid, validate on the random grid points, and
produce all figures.

Latency model (3 parameters, fitted on base-grid non-OOM points by
least squares on relative residuals):

    T(S, B) = sum_i max(F_i / F_eff, Q_i / BW_eff) + c

Energy model (3 parameters, non-negative least squares):

    E(S, B) = e_flop * FLOPs(S, B) + e_byte * Bytes(S, B) + E0

Outputs:
    results/theta.json        fitted parameters + fit diagnostics
    results/figures/*.png     all validation figures

Usage:  python calibrate.py [--results results]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, nnls

import equations as eq

REGIME_NAMES = {0: "launch-bound", 1: "memory-bound", 2: "compute-bound"}
COLORS = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868"}


def fit_latency(df_train, p0):
    """Fit {F_eff, BW_eff, c} by relative-error least squares (log-params)."""
    S = df_train["S"].to_numpy(float)
    B = df_train["B"].to_numpy(float)
    t = df_train["latency_s"].to_numpy(float)

    def resid(log_p):
        theta = {"F_eff": np.exp(log_p[0]), "BW_eff": np.exp(log_p[1]),
                 "c": np.exp(log_p[2])}
        pred = eq.latency(S, B, theta)
        return (pred - t) / t

    log_p0 = np.log([p0["F_eff"], p0["BW_eff"], p0["c"]])
    res = least_squares(resid, log_p0, method="lm", xtol=1e-12, ftol=1e-12)
    theta = {"F_eff": float(np.exp(res.x[0])), "BW_eff": float(np.exp(res.x[1])),
             "c": float(np.exp(res.x[2]))}
    return theta


def fit_energy(df_train):
    """NNLS fit of E = e_flop*FLOPs + e_byte*Bytes + E0."""
    S = df_train["S"].to_numpy(float)
    B = df_train["B"].to_numpy(float)
    E = df_train["energy_j"].to_numpy(float)
    X = np.stack([eq.flops(S, B), eq.bytes_moved(S, B),
                  np.ones_like(S, dtype=float)], axis=1)
    try:
        coef, _ = nnls(X, E)
    except Exception:
        coef, *_ = np.linalg.lstsq(X, E, rcond=None)
    return {"e_flop": float(coef[0]), "e_byte": float(coef[1]), "E0": float(coef[2])}


def rel_err(pred, meas):
    return (np.asarray(pred, float) - np.asarray(meas, float)) / np.asarray(meas, float)


def summarize(name, pred, meas, regimes=None):
    e = rel_err(pred, meas)
    line = (f"{name:28s} n={len(e):3d}  median|err|={np.median(np.abs(e))*100:6.2f}%  "
            f"max|err|={np.max(np.abs(e))*100:7.2f}%")
    if regimes is not None:
        for r, nm in REGIME_NAMES.items():
            m = regimes == r
            if m.sum():
                line += (f"\n    {nm:14s} n={int(m.sum()):3d}  "
                         f"median|err|={np.median(np.abs(e[m]))*100:6.2f}%")
    print(line)


def diagonal(ax, xs, ys, colors, markers, xlabel, ylabel, title, loglog=True):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    is_val = np.asarray(markers).astype(bool)
    for r in (0, 1, 2):
        m = (np.asarray(colors) == r) & ~is_val
        if m.sum():
            ax.scatter(xs[m], ys[m], s=18, c=COLORS[r],
                       label=REGIME_NAMES[r], alpha=0.8, edgecolor="none")
    if is_val.sum():
        ax.scatter(xs[is_val], ys[is_val], s=55, facecolor="none",
                   edgecolor="k", linewidths=1.1, label="validation")
    lo, hi = np.min([xs.min(), ys.min()]), np.max([xs.max(), ys.max()])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.results, "measurements.csv"))
    meta = {}
    meta_path = os.path.join(args.results, "run_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))

    ok = (df["status"] == "ok") & df["latency_s"].notna()
    train = df[ok & (df["is_validation"] == 0)].copy()
    val = df[ok & (df["is_validation"] == 1)].copy()
    print(f"configs: {len(df)} total, {len(train)} train (base), "
          f"{len(val)} validation, {(~ok).sum()} OOM/error")

    # ------------------------------------------------------------------ FLOPs
    if "flops_measured" in df and df["flops_measured"].notna().any():
        m = df["flops_measured"].to_numpy(float)
        p = df["flops_predicted"].to_numpy(float)
        summarize("FLOPs (measured vs formula)", p, m)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        diagonal(ax, m, p, np.zeros(len(m), int),
                 df["is_validation"].to_numpy().astype(bool),
                 "measured FLOPs (torch FlopCounterMode)", "FLOPs(S, B) formula",
                 "FLOPs: formula vs independent counter")
        fig.tight_layout(); fig.savefig(f"{figdir}/fig1_flops.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ memory
    if df["memory_bytes"].notna().any():
        mm = df.loc[df["memory_bytes"].notna()]
        pm = mm["memory_predicted"].to_numpy(float)
        rm = mm["memory_bytes"].to_numpy(float)
        summarize("peak memory (measured vs formula)", pm, rm)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
        diagonal(axes[0], rm, pm, np.zeros(len(mm), int),
                 mm["is_validation"].to_numpy().astype(bool),
                 "measured max_memory_allocated [bytes]",
                 "Memory(S, B) formula [bytes]", "Memory: formula vs measurement")
        # OOM map with predicted capacity frontier
        ax = axes[1]
        oom = df["status"] == "oom"
        ax.scatter(df.loc[~oom, "S"], df.loc[~oom, "B"], c="#55a868", s=22,
                   label="fits", edgecolor="none")
        if oom.any():
            ax.scatter(df.loc[oom, "S"], df.loc[oom, "B"], c="#c44e52", s=42,
                       marker="x", label="OOM")
            # empirical capacity: between the largest OK and smallest OOM fit
            cap = 0.5 * (df.loc[~oom, "memory_predicted"].max()
                          + df.loc[oom, "memory_predicted"].min())
            Sg = np.linspace(32, 512, 200)
            Bg = (cap - eq.PARAM_BYTES) / (44.0 * Sg**2)
            ax.plot(Sg, Bg, "k--", label=f"Memory = {cap/2**30:.1f} GiB")
        else:
            gpu_total = meta.get("gpu_total_bytes", 15.9e9)
            Sg = np.linspace(32, 512, 200)
            Bg = (gpu_total - eq.PARAM_BYTES) / (44.0 * Sg**2)
            ax.plot(Sg, Bg, "k--",
                    label=f"Memory = total VRAM ({gpu_total/2**30:.0f} GiB)")
        ax.set_yscale("log"); ax.set_ylim(0.7, 400)
        ax.set_xlabel("image size S"); ax.set_ylabel("batch size B")
        ax.set_title("OOM frontier: measured vs Memory(S,B)=capacity")
        ax.legend(fontsize=8); ax.grid(True, which="both", linestyle=":", alpha=0.4)
        fig.tight_layout(); fig.savefig(f"{figdir}/fig2_memory.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ latency
    theta = fit_latency(train, p0={"F_eff": 4e12, "BW_eff": 2e11, "c": 5e-5})
    print("\nlatency theta:", json.dumps(theta, indent=2))
    all_ok = df[ok]
    pred = eq.latency(all_ok["S"].to_numpy(float), all_ok["B"].to_numpy(float), theta)
    meas = all_ok["latency_s"].to_numpy(float)
    regs = eq.regime(all_ok["S"].to_numpy(float), all_ok["B"].to_numpy(float), theta)
    print("latency fit quality (relative error of T = sum max(F_i/F, Q_i/BW) + c):")
    summarize("  train (base grid)", eq.latency(train["S"], train["B"], theta),
              train["latency_s"].to_numpy(float))
    if len(val):
        summarize("  validation (random pts)", eq.latency(val["S"], val["B"], theta),
                  val["latency_s"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    diagonal(ax, meas, pred, regs,
             all_ok["is_validation"].to_numpy().astype(bool),
             "measured latency [s]", "Latency(S, B, theta) [s]",
             f"Latency: model vs measurement\nF_eff={theta['F_eff']/1e12:.2f} TFLOP/s, "
             f"BW_eff={theta['BW_eff']/1e9:.0f} GB/s, c={theta['c']*1e6:.0f} us")
    fig.tight_layout(); fig.savefig(f"{figdir}/fig3_latency_pred_vs_meas.png", dpi=150)
    plt.close(fig)

    # slices: T vs B (fixed S) and T vs S (fixed B)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    Bs = np.array(sorted(df["B"].unique()), float)
    for S in [64, 224, 512]:
        sub = all_ok[all_ok["S"] == S]
        if len(sub):
            axes[0].scatter(sub["B"], sub["latency_s"] * 1e3, s=20,
                            label=f"S={S} measured")
        axes[0].plot(Bs, eq.latency(S, Bs, theta) * 1e3, "-",
                     label=f"S={S} model")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size B"); axes[0].set_ylabel("latency [ms]")
    axes[0].set_title("latency vs batch size"); axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", linestyle=":", alpha=0.4)
    Ss = np.array(sorted(df["S"].unique()), float)
    for B in [1, 16, 256]:
        sub = all_ok[all_ok["B"] == B]
        if len(sub):
            axes[1].scatter(sub["S"], sub["latency_s"] * 1e3, s=20,
                            label=f"B={B} measured")
        axes[1].plot(Ss, eq.latency(Ss, B, theta) * 1e3, "-", label=f"B={B} model")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("image size S"); axes[1].set_ylabel("latency [ms]")
    axes[1].set_title("latency vs image size"); axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{figdir}/fig4_latency_slices.png", dpi=150)
    plt.close(fig)

    # 3D surface over (S, B)
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_subplot(projection="3d")
    Sg, Bg = np.meshgrid(np.geomspace(32, 512, 40), np.geomspace(1, 256, 40))
    Tg = eq.latency(Sg, Bg, theta)
    ax.plot_surface(np.log10(Sg), np.log10(Bg), np.log10(Tg), cmap="viridis",
                    alpha=0.75, linewidth=0, antialiased=True)
    sc = ax.scatter(np.log10(all_ok["S"]), np.log10(all_ok["B"]),
                    np.log10(meas), c=regs, cmap=matplotlib.colors.ListedColormap(
                        [COLORS[0], COLORS[1], COLORS[2]]), s=14, depthshade=False)
    ax.set_xlabel("log10 S"); ax.set_ylabel("log10 B"); ax.set_zlabel("log10 T [s]")
    ax.set_title("Latency surface Latency(S, B, theta) + measured points")
    fig.tight_layout(); fig.savefig(f"{figdir}/fig5_latency_surface.png", dpi=150)
    plt.close(fig)

    # regime map
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    Sg, Bg = np.meshgrid(np.linspace(32, 512, 300), np.geomspace(1, 256, 300))
    rg = eq.regime(Sg, Bg, theta)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([COLORS[0], COLORS[1], COLORS[2]])
    ax.pcolormesh(Sg, Bg, rg, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3),
                  alpha=0.35, shading="auto")
    for r in (0, 1, 2):
        m = regs == r
        if m.sum():
            ax.scatter(all_ok["S"].to_numpy(float)[m],
                       all_ok["B"].to_numpy(float)[m], c=COLORS[r], s=16,
                       label=REGIME_NAMES[r], edgecolor="none")
    ax.set_yscale("log"); ax.set_ylim(0.7, 400)
    ax.set_xlabel("image size S"); ax.set_ylabel("batch size B")
    ax.set_title("Predicted performance regimes over the measured grid")
    ax.legend(fontsize=8); ax.grid(True, which="both", linestyle=":", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/fig6_regime_map.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ energy
    theta_energy = None
    if "energy_j" in df and df["energy_j"].notna().sum() > 10:
        etrain = train[train["energy_j"].notna()]
        theta_energy = fit_energy(etrain)
        print("\nenergy theta_energy:", json.dumps(theta_energy, indent=2))
        eok = df[ok & df["energy_j"].notna()]
        epred = eq.energy(eok["S"].to_numpy(float), eok["B"].to_numpy(float),
                          theta_energy)
        emeas = eok["energy_j"].to_numpy(float)
        eregs = eq.regime(eok["S"].to_numpy(float), eok["B"].to_numpy(float), theta)
        summarize("energy train", eq.energy(etrain["S"], etrain["B"], theta_energy),
                 etrain["energy_j"].to_numpy(float))
        summarize("energy all (fit check)", epred, emeas)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
        diagonal(axes[0], emeas, epred, eregs,
                 eok["is_validation"].to_numpy().astype(bool),
                 "measured energy [J]", "Energy(S, B, theta_energy) [J]",
                 "Energy: model vs measurement")
        for S in [64, 224, 512]:
            sub = eok[eok["S"] == S]
            if len(sub):
                axes[1].scatter(sub["B"], sub["energy_j"], s=20, label=f"S={S} meas")
            axes[1].plot(Bs, eq.energy(S, Bs, theta_energy), "-", label=f"S={S} model")
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("batch size B"); axes[1].set_ylabel("energy [J]")
        axes[1].set_title("energy vs batch size"); axes[1].legend(fontsize=7)
        axes[1].grid(True, which="both", linestyle=":", alpha=0.4)
        fig.tight_layout(); fig.savefig(f"{figdir}/fig7_energy.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ theta.json
    out = {
        "latency": theta,
        "energy": theta_energy,
        "model": {
            "latency": "T = sum_i max(F_i/F_eff, Q_i/BW_eff) + c  (per-op roofline)",
            "energy": "E = e_flop*FLOPs + e_byte*Bytes + E0",
        },
        "gpu": meta.get("gpu_name"),
        "idle_power_w": meta.get("idle_power_w"),
        "notes": "fitted on base-grid non-OOM points only; "
                 "random grid points are validation",
    }
    with open(os.path.join(args.results, "theta.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.results}/theta.json and figures to {figdir}")


if __name__ == "__main__":
    main()
