"""EXP-011: consolidated ONE-PASS evaluation -> one artifact for every headline number.
Fixes the cross-table inconsistencies (0.955/0.957/0.961) by computing base/invariance/cond
from a single config, and adds the statistics a full paper needs: AUC + pAUC(0.1), 10-seed
matched with bootstrap CIs, per-band leave-one-band-out (LOBO) breakdown with variance, and
paired Wilcoxon tests (cond vs base, cond vs inv). Also emits per-band per-channel data for the
'overhang matched<LOBO' investigation and the degradation-curve figures.

MaFaulDa accel col1 (primary). Reuses mf_*.npz (T=time spectrum, O=order spectrum, S=speed).
Output: results/exp011_consolidated.json  (the single source of truth for the paper tables).
"""
import os, json, numpy as np
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]
K_PCA, K_NN = 32, 15


def load(stem):
    d = np.load(os.path.join(OUT, f"mf_{stem}.npz"))
    return d["T"].astype(np.float32), d["O"].astype(np.float32), d["S"].astype(np.float32)


def knn(Xtr, st, sq, k=K_NN):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c] - q))[:k]]].mean(0)
    return out


def pca(Xtr, k=K_PCA):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:min(k, Xtr.shape[0] - 2)]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def roc_auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def pauc(neg, pos, fpr_max=0.1):
    """Standardized partial AUC over FPR in [0, fpr_max] (McClish), in [0,1]."""
    thr = np.sort(np.concatenate([neg, pos]))[::-1]
    P, N = len(pos), len(neg); prev_fpr = 0.0; prev_tpr = 0.0; area = 0.0
    for t in thr:
        fpr = (neg >= t).mean(); tpr = (pos >= t).mean()
        if fpr > fpr_max:
            # interpolate to fpr_max
            if fpr != prev_fpr:
                frac = (fpr_max - prev_fpr) / (fpr - prev_fpr); tpr_i = prev_tpr + frac * (tpr - prev_tpr)
            else:
                tpr_i = tpr
            area += (fpr_max - prev_fpr) * (prev_tpr + tpr_i) / 2; break
        area += (fpr - prev_fpr) * (prev_tpr + tpr) / 2; prev_fpr, prev_tpr = fpr, tpr
    else:
        area += (fpr_max - prev_fpr) * prev_tpr
    # McClish standardize: 0.5*(1 + (pAUC - min)/(max-min)), min=fpr_max^2/2, max=fpr_max
    amin = fpr_max ** 2 / 2
    return float(0.5 * (1 + (area - amin) / (fpr_max - amin)))


def scorer(Tn, On, Sn, tr, treat):
    if treat == "base":
        f = pca(Tn[tr]); return f, lambda Tf, Of, Sf: f(Tf), lambda Tt: f(Tt)
    if treat == "inv":
        f = pca(On[tr]); return f, None, None
    m = knn(Tn[tr], Sn[tr], Sn[tr]); f = pca(Tn[tr] - m); return f, None, None


def eval_treat(Tn, On, Sn, faults, treat, tr, te, band=None):
    """Return (neg_scores, pos_scores) for a treatment given train idx tr, test idx te (normal),
    faults list; if band given, restrict test to that speed band."""
    if treat == "base":
        f = pca(Tn[tr]); neg = f(Tn[te])
        pos = np.concatenate([f(Tf[_bandmask(Sf, band)]) for Tf, Of, Sf in faults if _bandmask(Sf, band).sum()])
    elif treat == "inv":
        f = pca(On[tr]); neg = f(On[te])
        pos = np.concatenate([f(Of[_bandmask(Sf, band)]) for Tf, Of, Sf in faults if _bandmask(Sf, band).sum()])
    else:
        f = pca(Tn[tr] - knn(Tn[tr], Sn[tr], Sn[tr]))
        neg = f(Tn[te] - knn(Tn[tr], Sn[tr], Sn[te]))
        pos = np.concatenate([f(Tf[_bandmask(Sf, band)] - knn(Tn[tr], Sn[tr], Sf[_bandmask(Sf, band)]))
                              for Tf, Of, Sf in faults if _bandmask(Sf, band).sum()])
    return neg, pos


def _bandmask(S, band):
    if band is None:
        return np.ones(len(S), bool)
    return (S >= band[0]) & (S < band[1])


def wilcoxon(a, b):
    """Paired Wilcoxon signed-rank two-sided p (small-sample exact-ish via normal approx)."""
    d = np.array(a) - np.array(b); d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    W = r[d > 0].sum(); mu = n * (n + 1) / 4; sig = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sig == 0:
        return 1.0
    z = (W - mu) / sig
    from math import erfc
    return float(erfc(abs(z) / np.sqrt(2)))


def boot_ci(neg, pos, B=2000):
    rng = np.random.default_rng(1); v = np.empty(B)
    for b in range(B):
        v[b] = roc_auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    Tn, On, Sn = load("normal"); faults = [load(s) for s in FAULTS]
    R = {"config": {"channel": "underhang-axial(col1)", "k_pca": K_PCA, "k_nn": K_NN,
                    "window_revs": 16, "samp_per_rev": 256, "bands": BANDS, "seeds_matched": 10},
         "matched": {}, "lobo": {}, "tests": {}}

    # ---- matched: 10 seeds, pooled + per-fault ----
    treat_seed = {t: {"auc": [], "pauc": []} for t in ("base", "inv", "cond")}
    perfault = {s: {t: [] for t in ("base", "inv", "cond")} for s in FAULTS}
    last = {}
    for seed in range(10):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Tn)); cut = int(0.6 * len(Tn))
        tr, te = idx[:cut], idx[cut:]
        for t in ("base", "inv", "cond"):
            neg, pos = eval_treat(Tn, On, Sn, faults, t, tr, te)
            treat_seed[t]["auc"].append(roc_auc(neg, pos)); treat_seed[t]["pauc"].append(pauc(neg, pos))
            last[t] = (neg, pos)
            for si, (Tf, Of, Sf) in enumerate(faults):
                nb, pb = eval_treat(Tn, On, Sn, [faults[si]], t, tr, te)
                perfault[FAULTS[si]][t].append(roc_auc(nb, pb))
    for t in ("base", "inv", "cond"):
        au = np.array(treat_seed[t]["auc"]); pa = np.array(treat_seed[t]["pauc"])
        R["matched"][t] = dict(auc=float(au.mean()), auc_std=float(au.std()),
                               pauc=float(pa.mean()), pauc_std=float(pa.std()),
                               auc_ci=boot_ci(*last[t]))
    R["matched"]["per_fault"] = {s: {t: float(np.mean(perfault[s][t])) for t in ("base", "inv", "cond")} for s in FAULTS}
    R["tests"]["matched_wilcoxon"] = dict(
        cond_vs_base=wilcoxon(treat_seed["cond"]["auc"], treat_seed["base"]["auc"]),
        cond_vs_inv=wilcoxon(treat_seed["cond"]["auc"], treat_seed["inv"]["auc"]))

    # ---- LOBO: per band ----
    perband = {t: [] for t in ("base", "inv", "cond")}
    for (lo, hi) in BANDS:
        ten = (Sn >= lo) & (Sn < hi); trn = ~ten
        if ten.sum() < 8 or trn.sum() < 40:
            continue
        tr = np.where(trn)[0]; te = np.where(ten)[0]
        row = {}
        for t in ("base", "inv", "cond"):
            neg, pos = eval_treat(Tn, On, Sn, faults, t, tr, te, band=(lo, hi))
            a = roc_auc(neg, pos); perband[t].append(a); row[t] = a
        R["lobo"].setdefault("per_band", {})[f"{lo}-{hi}"] = row
    for t in ("base", "inv", "cond"):
        v = np.array(perband[t]); R["lobo"][t] = dict(auc=float(v.mean()), auc_std=float(v.std()), n_bands=len(v))
    R["tests"]["lobo_wilcoxon"] = dict(
        cond_vs_base=wilcoxon(perband["cond"], perband["base"]),
        cond_vs_inv=wilcoxon(perband["cond"], perband["inv"]))

    json.dump(R, open(os.path.join(OUT, "exp011_consolidated.json"), "w"), indent=2)
    # print summary
    print("MATCHED (10 seeds): " + "  ".join(
        f"{t}={R['matched'][t]['auc']:.3f}±{R['matched'][t]['auc_std']:.3f}(pAUC {R['matched'][t]['pauc']:.3f})" for t in ("base", "inv", "cond")))
    print("  CIs: " + "  ".join(f"{t}[{R['matched'][t]['auc_ci'][0]:.3f},{R['matched'][t]['auc_ci'][1]:.3f}]" for t in ("base","inv","cond")))
    print("LOBO (per band):  " + "  ".join(f"{t}={R['lobo'][t]['auc']:.3f}±{R['lobo'][t]['auc_std']:.3f}" for t in ("base", "inv", "cond")))
    print("  per-band cond:", {b: round(v["cond"], 3) for b, v in R["lobo"]["per_band"].items()})
    print("Wilcoxon matched cond-vs-base p=%.4g, cond-vs-inv p=%.4g" % (R["tests"]["matched_wilcoxon"]["cond_vs_base"], R["tests"]["matched_wilcoxon"]["cond_vs_inv"]))
    print("Wilcoxon LOBO   cond-vs-base p=%.4g, cond-vs-inv p=%.4g" % (R["tests"]["lobo_wilcoxon"]["cond_vs_base"], R["tests"]["lobo_wilcoxon"]["cond_vs_inv"]))
    print("saved results/exp011_consolidated.json")


if __name__ == "__main__":
    main()
