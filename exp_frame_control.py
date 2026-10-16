"""Decisive de-risk (pre-registered invariant P4) for the competing-equivariance direction.
Fuse DIFFERENT frame PAIRS with the SAME validated rule (max of held-out-healthy percentiles,
speed-conditioned per frame) and compare the fusion GAIN over the best single frame:
  (HZ, ORD)  -> the two most-different equivariances (resonance-fixed vs order-fixed): LEAD, expect big gain
  (HZ, REV)  -> same 'track-the-shaft' family-ish control: expect SMALL gain (< 0.02)
  (HZ, HZ)   -> trivial invariant: max(u,u)=u -> gain EXACTLY 0 (sanity)
PASS (physics framing holds): gain(HZ,ORD) large AND gain(HZ,REV) < 0.02 AND gain(HZ,HZ) == 0.
FAIL (generic ensembling): gain(HZ,REV) ~ gain(HZ,ORD) -> withdraw the physics framing.
MaFaulDa leave-one-speed-out, macro-AUC over 5 fault classes. Cached feats.npz only."""
import os, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import runpod_strongaccept_v2 as J
from sklearn.metrics import roc_auc_score


def ecdf(ts, q): ts = np.sort(ts); return np.searchsorted(ts, q, side="right") / (len(ts) + 1e-9)
def auc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def main():
    d = np.load("feats.npz")
    Sn, Rn, Sf, Ff = d["Nh_S"], d["Nh_R"], d["F_S"], d["F_FAULT"]
    FR = {"HZ": (d["Nh_HZ"], d["F_HZ"]), "REV": (d["Nh_REV"], d["F_REV"]), "ORD": (d["Nh_ORD"], d["F_ORD"])}
    SUB = int(os.environ.get("FAULT_SUB", "8000"))
    if 0 < SUB < len(Sf):
        rng = np.random.default_rng(0); keep = []
        for c in np.unique(Ff):
            ii = np.where(Ff == c)[0]; keep.append(rng.choice(ii, min(len(ii), SUB // len(np.unique(Ff))), replace=False))
        sel = np.sort(np.concatenate(keep)); Sf, Ff = Sf[sel], Ff[sel]
        FR = {k: (nh, ff[sel]) for k, (nh, ff) in FR.items()}
    J.log(f"frames {list(FR)}; healthy {len(Sn)} fault {len(Sf)}")
    recs = np.unique(Rn); faults = np.unique(Ff)

    # precompute per-fold per-frame held-out-healthy percentiles; evaluate arbitrary frame GROUPS
    groups = [("HZ", "ORD"), ("HZ", "REV"), ("REV", "ORD"), ("HZ", "REV", "ORD")]
    frames_needed = ["HZ", "REV", "ORD"]
    single = {f: [] for f in frames_needed}
    fused = {g: [] for g in groups}
    for ri, rid in enumerate(recs):
        te = Rn == rid; tr = Rn != rid
        if te.sum() < 5 or tr.sum() < 40: continue
        if ri % 10 == 0: J.log(f"  fold {ri}")
        uH = {}; uF = {}
        for f in frames_needed:
            Xn, Xf = FR[f]
            mtr = J.knn_mean(Xn[tr], Sn[tr], Sn[tr]); g = J.pca(Xn[tr] - mtr)
            steH = g(Xn[te] - J.knn_mean(Xn[tr], Sn[tr], Sn[te])); sF = g(Xf - J.knn_mean(Xn[tr], Sn[tr], Sf))
            uH[f] = ecdf(steH, steH); uF[f] = ecdf(steH, sF)
        for c in faults:
            fm = Ff == c
            if fm.sum() < 3: continue
            for f in frames_needed:
                single[f].append(auc(uH[f], uF[f][fm]))
            for g in groups:
                fa = np.max([uH[f] for f in g], axis=0); fb = np.max([uF[f][fm] for f in g], axis=0)
                fused[g].append(auc(fa, fb))
    sm = {f: round(float(np.mean(v)), 4) for f, v in single.items()}
    best_rev = sm["REV"]
    res = {"single_macro_auc": sm, "groups": {}}
    for g in groups:
        fu = round(float(np.mean(fused[g])), 4); best = max(sm[f] for f in g)
        res["groups"]["+".join(g)] = {"fused": fu, "best_member": round(best, 4),
                                      "gain_over_best_member": round(fu - best, 4),
                                      "gain_over_REV": round(fu - best_rev, 4)}
    # method contribution test: does any fusion beat REV-alone (the strongest single conditioned frame)?
    beats_rev = {k: v["gain_over_REV"] for k, v in res["groups"].items()}
    res["best_fusion_vs_REV"] = round(max(beats_rev.values()), 4)
    res["method_contribution"] = "YES (a fusion beats REV)" if res["best_fusion_vs_REV"] > 0.01 else "NO (REV-alone >= any fusion)"
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/frame_control.json", "w"), indent=2)
    J.log("single: " + json.dumps(sm))
    for k, v in res["groups"].items(): J.log(f"  {k}: fused {v['fused']} (vs best-member {v['best_member']}: {v['gain_over_best_member']:+}) (vs REV: {v['gain_over_REV']:+})")
    J.log("method contribution (fusion beats REV-alone?): " + res["method_contribution"])
    print("[ctrl] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
