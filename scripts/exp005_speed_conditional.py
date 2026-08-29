"""EXP-005: SPEED-CONDITIONAL normal model. The consistent finding across EXP-002..004 is
that unconditional detectors let fault energy hide in the speed-induced spread of the normal
class. Idea: model the normal feature manifold CONDITIONED on the measured tacho speed, and
score deviation from that speed-conditional normal. We have a continuous physical covariate
(speed) that DCASE domain-shift methods lack.

Implementation: per-dim smooth speed trend m_j(speed) (poly deg 3) fit on normal-train;
residual r = X - M(speed); PCA on normal residuals; score = recon error of test residual
(each test point detrended by its OWN speed). Compare to unconditional accel-PCA.

Two protocols:
  (1) matched-range: normal train spans all speeds (60/40). Does conditioning tighten normal
      and raise fault AUC?
  (2) leave-one-band-out (interpolation): train normal on 5 bands, test held-out band's
      normal vs fault -- fairer cross-speed than single-band extrapolation.
Reuses cache_ch1_r16_s256.npz (accel col1, 'time' mode). Bootstrap CIs.
"""
import os, numpy as np
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:min(k, X.shape[0] - 2)])


def pca_err(m, X):
    Xs = (X - m["mu"]) / m["sd"]
    return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def fit_speed_trend(X, s, deg=3):
    """Per-dim polynomial mean vs speed. Returns coeffs (deg+1, D) and eval fn."""
    V = np.vander(s, deg + 1)                       # (N, deg+1)
    coef, *_ = np.linalg.lstsq(V, X, rcond=None)    # (deg+1, D)
    return coef


def apply_trend(coef, s, deg=3):
    return np.vander(s, deg + 1) @ coef


def auc(neg, pos):
    allv = np.concatenate([neg, pos]); _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0; r = avg[inv]; n1 = len(pos)
    return (r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1)


def boot(neg, pos, B=1000):
    rng = np.random.default_rng(1); v = np.empty(B)
    for b in range(B):
        v[b] = auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))])
    return np.percentile(v, [2.5, 97.5])


def main():
    cache = np.load(os.path.join(OUT, "cache_ch1_r16_s256.npz"), allow_pickle=True)["obj"].item()
    Xn, sn = cache["normal"]["time"]; Xf, sf = cache["fault"]["time"]
    K, DEG = 32, 3

    print("### (1) MATCHED-RANGE: normal train spans all speeds (60/40) ###")
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xn)); cut = int(0.6 * len(Xn))
    tr, te = idx[:cut], idx[cut:]
    # unconditional
    mu = pca_fit(Xn[tr], K)
    a0 = auc(pca_err(mu, Xn[te]), pca_err(mu, Xf)); lo0, hi0 = boot(pca_err(mu, Xn[te]), pca_err(mu, Xf))
    # speed-conditional
    coef = fit_speed_trend(Xn[tr], sn[tr], DEG)
    Rtr = Xn[tr] - apply_trend(coef, sn[tr], DEG)
    mc = pca_fit(Rtr, K)
    s_n = pca_err(mc, Xn[te] - apply_trend(coef, sn[te], DEG))
    s_f = pca_err(mc, Xf - apply_trend(coef, sf, DEG))
    a1 = auc(s_n, s_f); lo1, hi1 = boot(s_n, s_f)
    print(f"[unconditional PCA ] AUC={a0:.3f} [{lo0:.3f},{hi0:.3f}]")
    print(f"[speed-conditional ] AUC={a1:.3f} [{lo1:.3f},{hi1:.3f}]")
    # paired bootstrap diff
    rng = np.random.default_rng(5)
    e0n, e0f = pca_err(mu, Xn[te]), pca_err(mu, Xf)
    d = []
    for _ in range(2000):
        ni = rng.integers(0, len(s_n), len(s_n)); fi = rng.integers(0, len(s_f), len(s_f))
        d.append(auc(s_n[ni], s_f[fi]) - auc(e0n[ni], e0f[fi]))
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    print(f"  cond - uncond paired diff = {np.mean(d):+.3f} CI[{dlo:+.3f},{dhi:+.3f}] "
          f"({'SIGNIFICANT' if dlo>0 else 'ns'})")

    print("\n### (2) LEAVE-ONE-BAND-OUT (interpolation): train 5 bands, test held-out band ###")
    bands = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]
    unc, con = [], []
    for bi, (lo, hi) in enumerate(bands):
        te_mask_n = (sn >= lo) & (sn < hi); tr_mask_n = ~te_mask_n
        te_mask_f = (sf >= lo) & (sf < hi)
        if te_mask_n.sum() < 8 or te_mask_f.sum() < 8 or tr_mask_n.sum() < K + 5:
            continue
        Xtr, str_ = Xn[tr_mask_n], sn[tr_mask_n]
        # unconditional
        mu = pca_fit(Xtr, K)
        au = auc(pca_err(mu, Xn[te_mask_n]), pca_err(mu, Xf[te_mask_f]))
        # conditional
        coef = fit_speed_trend(Xtr, str_, DEG)
        mc = pca_fit(Xtr - apply_trend(coef, str_, DEG), K)
        sc_n = pca_err(mc, Xn[te_mask_n] - apply_trend(coef, sn[te_mask_n], DEG))
        sc_f = pca_err(mc, Xf[te_mask_f] - apply_trend(coef, sf[te_mask_f], DEG))
        ac = auc(sc_n, sc_f)
        unc.append(au); con.append(ac)
        print(f"  held-out {lo}-{hi:>2} Hz: uncond={au:.3f}  cond={ac:.3f}  (n_te_n={te_mask_n.sum()}, n_te_f={te_mask_f.sum()})")
    if unc:
        print(f"  MEAN: uncond={np.mean(unc):.3f}  cond={np.mean(con):.3f}  "
              f"diff={np.mean(con)-np.mean(unc):+.3f}")


if __name__ == "__main__":
    main()
