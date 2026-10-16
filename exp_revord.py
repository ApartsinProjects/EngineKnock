"""Does adding the ORDER frame to the strong REV-conditioned detector beat REV alone?
Decides method-contribution vs mechanism-only. REV and ORD both 2049-dim (memory-light)."""
import os, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import runpod_strongaccept_v2 as J
from sklearn.metrics import roc_auc_score
def ecdf(ts, q): ts = np.sort(ts); return np.searchsorted(ts, q, side="right") / (len(ts) + 1e-9)
def auc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
d = np.load("feats.npz"); Sn, Rn, Sf, Ff = d["Nh_S"], d["Nh_R"], d["F_S"], d["F_FAULT"]
FR = {"REV": (d["Nh_REV"], d["F_REV"]), "ORD": (d["Nh_ORD"], d["F_ORD"])}
SUB = int(os.environ.get("FAULT_SUB", "4000")); rng = np.random.default_rng(0); keep = []
for c in np.unique(Ff):
    ii = np.where(Ff == c)[0]; keep.append(rng.choice(ii, min(len(ii), SUB // len(np.unique(Ff))), replace=False))
sel = np.sort(np.concatenate(keep)); Sf, Ff = Sf[sel], Ff[sel]; FR = {k: (nh, ff[sel]) for k, (nh, ff) in FR.items()}
recs = np.unique(Rn); faults = np.unique(Ff); S = {"REV": [], "ORD": []}; F2 = []
for ri, rid in enumerate(recs):
    te = Rn == rid; tr = Rn != rid
    if te.sum() < 5 or tr.sum() < 40: continue
    uH = {}; uF = {}
    for f in ("REV", "ORD"):
        Xn, Xf = FR[f]; mtr = J.knn_mean(Xn[tr], Sn[tr], Sn[tr]); g = J.pca(Xn[tr] - mtr)
        steH = g(Xn[te] - J.knn_mean(Xn[tr], Sn[tr], Sn[te])); sF = g(Xf - J.knn_mean(Xn[tr], Sn[tr], Sf))
        uH[f] = ecdf(steH, steH); uF[f] = ecdf(steH, sF)
    for c in faults:
        fm = Ff == c
        if fm.sum() < 3: continue
        S["REV"].append(auc(uH["REV"], uF["REV"][fm])); S["ORD"].append(auc(uH["ORD"], uF["ORD"][fm]))
        F2.append(auc(np.maximum(uH["REV"], uH["ORD"]), np.maximum(uF["REV"][fm], uF["ORD"][fm])))
rev = float(np.mean(S["REV"])); ordd = float(np.mean(S["ORD"])); fus = float(np.mean(F2))
res = {"REV": round(rev, 4), "ORD": round(ordd, 4), "REV+ORD_fused": round(fus, 4),
       "gain_over_REV": round(fus - rev, 4),
       "method_contribution": "YES" if fus - rev > 0.01 else "NO (REV-alone >= REV+ORD)"}
json.dump(res, open("results/frame_revord.json", "w"), indent=2)
print(f"[revord] REV {rev:.4f} ORD {ordd:.4f} REV+ORD {fus:.4f} | gain over REV {fus-rev:+.4f} -> {res['method_contribution']}", flush=True)
print("[revord] === DONE ===", flush=True)
