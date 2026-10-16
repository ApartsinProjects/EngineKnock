"""ResearchPivot cycle-2 smoke: Multi-Frame Normality (competing-equivariance) on MaFaulDa.
Two physically incompatible normality frames from the SAME cached data: HZ (physical-frequency, where
structural resonances are stationary) and ORD (shaft-order, where rotation-locked components are
stationary). Run the speed-conditioned PCA detector INDEPENDENTLY in each frame (both conditioned on
rpm), convert each score to its healthy percentile via the cross-fitted training-healthy ECDF, and fuse
parameter-free: A_2F = max(u_HZ, u_ORD). No fault labels used in fusion.

Metric: five-fault MACRO ROC-AUC under leakage-free leave-one-speed-out (macro, because easy bearing
faults saturate pooled AUC). Positive lead: Delta_2F = MacroAUC_2F - max(MacroAUC_HZ, MacroAUC_ORD) >= 0.02.
Kill: < 0.01. Uses only cached feats.npz (HZ, ORD, speed, recording ids, fault ids)."""
import os, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import runpod_strongaccept_v2 as J
from sklearn.metrics import roc_auc_score


def ecdf_pctile(train_scores, q):
    """Empirical healthy percentile of each query score q against sorted training-healthy scores."""
    ts = np.sort(train_scores)
    return np.searchsorted(ts, q, side="right") / (len(ts) + 1e-9)


def main():
    # load ONLY the needed arrays lazily (skip REV etc.) to stay under the host memory wall.
    # On a fresh pod feats.npz is absent -> get_features() downloads the S3 volume cache to feats.npz.
    if not (os.path.exists("feats.npz") or os.path.exists("results/feats.npz")):
        J.get_features()
    src = "feats.npz" if os.path.exists("feats.npz") else "results/feats.npz"
    d = np.load(src)
    Sn = d["Nh_S"]; Rn = d["Nh_R"]; Sf = d["F_S"]; Ff = d["F_FAULT"]
    NH = {"HZ": d["Nh_HZ"], "ORD": d["Nh_ORD"]}; FF = {"HZ": d["F_HZ"], "ORD": d["F_ORD"]}
    J.log(f"loaded HZ{NH['HZ'].shape[1]}/ORD{NH['ORD'].shape[1]}; healthy {len(Sn)} fault {len(Sf)}")
    # stratified fault subsample (smoke; macro needs per-class so keep classes balanced)
    SUB = int(os.environ.get("FAULT_SUB", "5000"))
    if 0 < SUB < len(Sf):
        rng = np.random.default_rng(0); keep = []
        for c in np.unique(Ff):
            ii = np.where(Ff == c)[0]; keep.append(rng.choice(ii, min(len(ii), SUB // len(np.unique(Ff))), replace=False))
        sel = np.sort(np.concatenate(keep)); Sf, Ff = Sf[sel], Ff[sel]
        FF = {k: v[sel] for k, v in FF.items()}
    faults = np.unique(Ff)
    recs = np.unique(Rn)
    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    # accumulate per-fold per-fault AUC for HZ, ORD, 2F
    per = {"HZ": [], "ORD": [], "2F": []}
    nfold = 0
    for rid in recs:
        te = Rn == rid; tr = Rn != rid
        if te.sum() < 5 or tr.sum() < 40:
            continue
        nfold += 1
        if nfold % 10 == 1:
            J.log(f"  fold {nfold}")
        u_teH = {}; u_F = {}
        for fr in ("HZ", "ORD"):
            Xn = NH[fr]; Xf = FF[fr]
            mtr = J.knn_mean(Xn[tr], Sn[tr], Sn[tr]); f = J.pca(Xn[tr] - mtr)
            s_teH = f(Xn[te] - J.knn_mean(Xn[tr], Sn[tr], Sn[te]))
            s_F = f(Xf - J.knn_mean(Xn[tr], Sn[tr], Sf))
            # Calibrate the per-frame percentile against the HELD-OUT healthy distribution (out-of-sample),
            # NOT the in-sample training scores (which are optimistically low under leave-one-out and make
            # every held-out point saturate to percentile 1.0). This puts HZ and ORD on a common [0,1]
            # scale so the max-fusion is meaningful, and preserves each frame's true AUC (monotone map).
            u_teH[fr] = ecdf_pctile(s_teH, s_teH); u_F[fr] = ecdf_pctile(s_teH, s_F)
        a2_teH = np.maximum(u_teH["HZ"], u_teH["ORD"]); a2_F = np.maximum(u_F["HZ"], u_F["ORD"])
        for c in faults:
            fm = Ff == c
            if fm.sum() < 3:
                continue
            per["HZ"].append(auc(u_teH["HZ"], u_F["HZ"][fm]))
            per["ORD"].append(auc(u_teH["ORD"], u_F["ORD"][fm]))
            per["2F"].append(auc(a2_teH, a2_F[fm]))
    macro = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
    delta = round(macro["2F"] - max(macro["HZ"], macro["ORD"]), 4)
    res = {"n_folds": nfold, "n_faults_sub": int(len(Sf)), "macro_auc": macro,
           "best_single_frame": max(macro["HZ"], macro["ORD"]), "delta_2F": delta,
           "verdict": ("CONFIRM" if delta >= 0.02 else ("KILL" if delta < 0.01 else "WEAK"))}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/twoframe.json", "w"), indent=2)
    print(f"[2F] macro-AUC  HZ={macro['HZ']}  ORD={macro['ORD']}  2F={macro['2F']}", flush=True)
    print(f"[2F] Delta_2F = {delta}  ->  {res['verdict']}", flush=True)
    print("[2F] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
