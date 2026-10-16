"""ResearchPivot cycle-1 smoke: Healthy-Only Benchmark Admissibility (placebo) test on Ottawa.
NO fault files used. Ottawa healthy = 4 speed trajectories (A/B/C/D) x 3 trials (1/2/3). Within a
trajectory the 3 trials share the same speed sweep, so speed is matched by construction; conditioning on
per-window RPM removes any residual speed effect, and L2-normalizing each spectrum removes level. Any
remaining separability between two healthy trials is pure ACQUISITION-SESSION identity.

Per trajectory, rotate: train PCA on one trial, score the other two, call one 'normal' and one
'pseudo-anomaly', AUC, take max(AUC,1-AUC) (direction of the session fingerprint is irrelevant).
3 rotations x 4 trajectories = 12 all-healthy placebo tests. PlaceboAUC = median.
Positive lead (confirms pivot): PlaceboAUC >= 0.80 under BOTH L2-norm and RPM conditioning.
Report raw / L2-only / L2+RPM so the effect of each control is visible."""
import os, glob, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import ottawa_run as O
from sklearn.metrics import roc_auc_score

d = np.load(O.FEAT)
REV = d["REV"]; S = d["S"]; CLS = d["CLS"]; REC = d["REC"]
names = [os.path.basename(p) for p in sorted(glob.glob("data/ottawa/*.mat"))]  # REC index -> filename


def auc(a, b): return float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
def l2(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def placebo(feat, cond_rpm):
    """feat: (N,d) healthy features aligned to REC. Return list of 12 max(AUC,1-AUC) placebo values."""
    vals = []
    for prof in ["A", "B", "C", "D"]:
        # the 3 healthy trials for this trajectory
        trec = [ri for ri, nm in enumerate(names) if nm.startswith(f"H-{prof}-")]
        if len(trec) < 3:
            continue
        for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            ra, rb, rc = trec[a], trec[b], trec[c]
            ma = (REC == ra); mb = (REC == rb); mc = (REC == rc)
            Xa, Sa = feat[ma], S[ma]; Xb, Sb = feat[mb], S[mb]; Xc, Sc = feat[mc], S[mc]
            if cond_rpm:
                mtr = O.knn_mean(Xa, Sa, Sa)
                f = O.pca(Xa - mtr)
                sb = f(Xb - O.knn_mean(Xa, Sa, Sb)); sc = f(Xc - O.knn_mean(Xa, Sa, Sc))
            else:
                f = O.pca(Xa); sb = f(Xb); sc = f(Xc)
            a_ = auc(sb, sc); vals.append(max(a_, 1 - a_))
    return vals


res = {}
for tag, feat, cond in [("raw", REV, False), ("L2norm", l2(REV), False), ("L2norm+RPM", l2(REV), True)]:
    Hmask = CLS == "H"
    # restrict features/arrays to healthy only for clarity (placebo uses no faults)
    v = placebo(feat, cond)
    res[tag] = {"placebo_auc_median": round(float(np.median(v)), 3),
                "n_tests": len(v), "ge_0.70": int(np.sum(np.array(v) >= 0.70)),
                "per_test": [round(x, 3) for x in v]}
    print(f"[placebo] {tag}: PlaceboAUC(median)={res[tag]['placebo_auc_median']} "
          f"| {res[tag]['ge_0.70']}/{len(v)} >= 0.70", flush=True)

os.makedirs("results", exist_ok=True)
json.dump(res, open("results/placebo_ottawa.json", "w"), indent=2)
print("[placebo] verdict:", "CONFIRM pivot" if res["L2norm+RPM"]["placebo_auc_median"] >= 0.80
      else ("KILL" if res["L2norm+RPM"]["placebo_auc_median"] < 0.60 else "WEAK"), flush=True)
print("[placebo] === DONE ===", flush=True)
