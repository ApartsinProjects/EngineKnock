"""Confirmatory per-fault frame-blindness numbers for the paper (the MECHANISM that explains why
order tracking / REMOVE is the worst treatment: it is blind to structural faults).
For each representation frame (HZ physical-frequency / REV revolution-normalized / ORD order-tracked),
speed-conditioned normal-only detector, leakage-free leave-one-speed-out: per-fault ROC-AUC averaged
over the held-out recordings, with cluster-bootstrap 95% CI over recordings.
Memory-safe: processes ONE frame at a time (loads only that frame's arrays, frees before the next),
so the 4097-dim HZ array never coexists with the others. One co-computed artifact."""
import os, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import runpod_strongaccept_v2 as J
from sklearn.metrics import roc_auc_score

FAULTS = {0: "horizontal misalignment", 1: "vertical misalignment", 2: "imbalance",
          3: "underhang bearing", 4: "overhang bearing"}
STRUCT = {0, 1}  # alignment/structural (resonance-type); the rest are rotational/order-locked


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def boot_ci(vals, n=2000, seed=1):
    v = np.asarray(vals); rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    d = np.load("feats.npz")
    Sn, Rn, Sf, Ff = d["Nh_S"], d["Nh_R"], d["F_S"], d["F_FAULT"]
    SUB = int(os.environ.get("FAULT_SUB", "8000"))
    if 0 < SUB < len(Sf):
        rng = np.random.default_rng(0); keep = []
        for c in np.unique(Ff):
            ii = np.where(Ff == c)[0]; keep.append(rng.choice(ii, min(len(ii), SUB // len(np.unique(Ff))), replace=False))
        selmask = np.sort(np.concatenate(keep))
    else:
        selmask = np.arange(len(Sf))
    Sf, Ff = Sf[selmask], Ff[selmask]
    recs = np.unique(Rn); faults = sorted(np.unique(Ff).tolist())
    J.log(f"healthy {len(Sn)}/{len(recs)} recs; fault {len(Sf)} ({SUB=})")

    res = {"frames": {}, "config": {"protocol": "leakage-free LOSO, speed-conditioned, per-fault ROC-AUC",
                                     "fault_subsample": int(SUB), "ci": "cluster-bootstrap over recordings"}}
    for frame, (nhk, fk) in [("HZ", ("Nh_HZ", "F_HZ")), ("REV", ("Nh_REV", "F_REV")), ("ORD", ("Nh_ORD", "F_ORD"))]:
        Xn = d[nhk]; Xf = d[fk][selmask]
        J.log(f"frame {frame}: dim {Xn.shape[1]}")
        perfault = {c: [] for c in faults}
        for ri, rid in enumerate(recs):
            te = Rn == rid; tr = Rn != rid
            if te.sum() < 5 or tr.sum() < 40:
                continue
            mtr = J.knn_mean(Xn[tr], Sn[tr], Sn[tr]); g = J.pca(Xn[tr] - mtr)
            sH = g(Xn[te] - J.knn_mean(Xn[tr], Sn[tr], Sn[te]))
            sF = g(Xf - J.knn_mean(Xn[tr], Sn[tr], Sf))
            for c in faults:
                fm = Ff == c
                if fm.sum() >= 3:
                    perfault[c].append(auc(sH, sF[fm]))
        res["frames"][frame] = {FAULTS[c]: {"auc": round(float(np.mean(perfault[c])), 3),
                                            "ci95": boot_ci(perfault[c]), "n_folds": len(perfault[c])}
                                for c in faults}
        del Xn, Xf  # free before the next frame
        J.log(f"  {frame}: " + json.dumps({FAULTS[c]: res['frames'][frame][FAULTS[c]]['auc'] for c in faults}))

    # structural vs rotational summary per frame
    res["summary"] = {}
    for frame in ("HZ", "REV", "ORD"):
        st = np.mean([res["frames"][frame][FAULTS[c]]["auc"] for c in faults if c in STRUCT])
        ro = np.mean([res["frames"][frame][FAULTS[c]]["auc"] for c in faults if c not in STRUCT])
        res["summary"][frame] = {"structural_mean_auc": round(float(st), 3), "rotational_mean_auc": round(float(ro), 3)}
    json.dump(res, open("results/frameblind.json", "w"), indent=2)
    J.log("summary: " + json.dumps(res["summary"]))
    J.log("=== DONE ===")


if __name__ == "__main__":
    main()
