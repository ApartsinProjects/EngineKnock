"""Driver: run the three strengthening experiments on RunPod, cache-hitting feats.npz from the volume.
Runs to completion independently of the local session. Results land in results/ (fetched back)."""
import os, sys, traceback
os.environ.setdefault("FAULT_SUB", "0")      # 0 = use ALL faults (pod has RAM)
os.environ.setdefault("TT_SEEDS", "10")      # 10 seeds for threshold transfer
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # CPU numpy/sklearn/torch

def run(name, fn):
    print(f"\n==== {name} ====", flush=True)
    try:
        fn(); print(f"==== {name} OK ====", flush=True)
    except Exception:
        print(f"==== {name} FAILED ====\n{traceback.format_exc()}", flush=True)

import runpod_recipe, exp_latency, exp_condae
run("recipe (support-aware + ignore_qnorm + threshold transfer + CIs)", runpod_recipe.main)
run("latency (regime early-warning)", exp_latency.main)
run("condae (conditional-AE baseline)", exp_condae.main)
print("\n[driver] === ALL DONE ===", flush=True)
