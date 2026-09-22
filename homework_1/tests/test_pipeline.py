"""Pipeline self-test without a GPU.

Generates synthetic measurements from the equations with known ground-truth
theta (T4-like) plus noise, then checks that calibrate.py's estimator
recovers the parameters. Run:  python tests/test_pipeline.py
"""
import json, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import equations as eq
from measure import make_grid

TRUE_THETA = {"F_eff": 3.6e12, "BW_eff": 2.3e11, "c": 6e-5}
TRUE_ETHETA = {"e_flop": 5.0e-14, "e_byte": 2.0e-10, "E0": 0.002}
CAPACITY = 1.5e9   # small fake VRAM so that OOM cells exist in the test
NOISE = 0.05

def main():
    rng = np.random.default_rng(0)
    configs, rs, rb = make_grid(0)
    rows = []
    for (S, B) in configs:
        S_, B_ = float(S), float(B)
        mem = eq.memory(S_, B_)
        rec = dict(S=S, B=B, is_validation=int(S not in (32,64,128,224,256,384,512)
                                               or B not in (1,2,4,8,16,32,64,128,256)),
                   flops_measured=eq.flops(S_, B_), flops_predicted=eq.flops(S_, B_),
                   memory_predicted=mem)
        if mem > CAPACITY:
            rec.update(status="oom", latency_s=np.nan, memory_bytes=np.nan,
                       energy_j=np.nan)
        else:
            t = eq.latency(S_, B_, TRUE_THETA) * float(np.exp(rng.normal(0, NOISE)))
            e = eq.energy(S_, B_, TRUE_ETHETA) * float(np.exp(rng.normal(0, NOISE)))
            rec.update(status="ok", latency_s=t, memory_bytes=mem * 1.03, energy_j=e)
        rows.append(rec)
    df = pd.DataFrame(rows)

    out = tempfile.mkdtemp(prefix="synth_")
    df.to_csv(os.path.join(out, "measurements.csv"), index=False)
    json.dump({"gpu_name": "SYNTHETIC-T4", "gpu_total_bytes": int(CAPACITY),
               "idle_power_w": 11.0, "torch": "test", "seed": 0,
               "rand_sizes": rs, "rand_batches": rb},
              open(os.path.join(out, "run_meta.json"), "w"))

    r = subprocess.run([sys.executable, "calibrate.py", "--results", out],
                       cwd=os.path.join(os.path.dirname(__file__), ".."),
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])

    theta = json.load(open(os.path.join(out, "theta.json")))["latency"]
    print("\nrecovered vs true:")
    for k in TRUE_THETA:
        rel = theta[k] / TRUE_THETA[k] - 1
        print(f"  {k:8s} true={TRUE_THETA[k]:.4e} fitted={theta[k]:.4e} rel={rel:+.1%}")
        assert abs(rel) < 0.25, f"{k} not recovered"

    figs = sorted(os.listdir(os.path.join(out, "figures")))
    print("figures:", figs)
    assert len(figs) >= 6
    print("\nPIPELINE TEST PASSED")

if __name__ == "__main__":
    main()
