"""EXP-004: unsupervised CROSS-MODAL anomaly detection on MaFaulDa (accel <-> mic).

Thesis: a normal-only learned map between accelerometer and microphone spectra captures the
healthy vibro-acoustic coupling; faults break it, raising the cross-modal residual. The map
may be more speed-invariant than either spectrum alone (a speed change moves both modalities
together), so cross-modal residual should generalize across speed where single-modality
detectors collapse (the regime that killed order-sync in EXP-003).

Detectors compared (all unsupervised, normal-only fit), one pass, construct-matched windows:
  accel_pca   : PCA recon error on accel spectrum          (single modality)
  mic_pca     : PCA recon error on mic spectrum            (single modality)
  concat_pca  : PCA recon error on [accel|mic]             (more features, not agreement)
  xmodal_a2m  : ridge map accel->mic, score=||mic - A accel||   (agreement)
  xmodal_m2a  : ridge map mic->accel, score=||accel - B mic||   (agreement)
  xmodal_sym  : mean of the two cross-modal residuals
Control:
  xmodal_shuffle : a2m residual but test mic rows permuted -> real coupling destroyed.

Channels: accel = underhang axial (col1); mic = col7. Window = 16 revs, 256 samp/rev.
Reports matched-speed AUC and confound-free cross-speed mean AUC, with bootstrap CIs.
"""
import io, os, zipfile
import numpy as np

FS = 50_000
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
ACC_COL, MIC_COL, REVS, SPER = 1, 7, 16, 256


def tacho_rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def win_spectra(acc, mic, rises, revs=REVS, sper=SPER):
    """One fixed-time window per `revs` revolutions; return (accel_logspec, mic_logspec)."""
    if len(rises) < revs + 1:
        return
    L = revs * sper
    nb = (len(rises) - 1) // revs
    win = np.hanning(L)
    for b in range(nb):
        p0, p1 = rises[b * revs], rises[(b + 1) * revs]
        if p1 <= p0 or p1 > len(acc):
            continue
        def spec(sig):
            seg = sig[p0:p1].astype(float)
            w = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
            w = w - w.mean()
            return np.log1p(np.abs(np.fft.rfft(w * win)))
        yield spec(acc), spec(mic)


def build_cache(zip_path):
    A, M, S = [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        for n in sorted(x for x in zf.namelist() if x.lower().endswith(".csv")):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")
            rises = tacho_rises(arr[:, 0])
            frot = FS / np.median(np.diff(rises)) if len(rises) > 2 else np.nan
            for a, m in win_spectra(arr[:, ACC_COL], arr[:, MIC_COL], rises):
                A.append(a); M.append(m); S.append(frot)
    return np.asarray(A), np.asarray(M), np.asarray(S)


def get_cache():
    p = os.path.join(OUT, "cache_xmodal_ch1mic7_r16_s256.npz")
    if os.path.exists(p):
        d = np.load(p)
        return {k: (d[f"{k}_A"], d[f"{k}_M"], d[f"{k}_S"]) for k in ("normal", "fault")}
    out = {}
    for zp, lab in [("normal.zip", "normal"), ("horizontal-misalignment.zip", "fault")]:
        out[lab] = build_cache(os.path.join(DATA, zp))
    np.savez_compressed(p, **{f"{lab}_{x}": arr
                              for lab in out for x, arr in zip("AMS", out[lab])})
    return out


# ---------- detectors ----------
def _std(X, mu, sd): return (X - mu) / sd


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd(_std(X, mu, sd), full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:min(k, X.shape[0] - 2)])


def pca_err(m, X):
    Xs = _std(X, m["mu"], m["sd"])
    return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def ridge_fit(Xin, Yout, k=48, lam=1.0):
    """Reduce both sides via PCA(normal), fit ridge in reduced space (stable, cheap)."""
    pin = pca_fit(Xin, k=k); pout = pca_fit(Yout, k=k)
    Zi = _std(Xin, pin["mu"], pin["sd"]) @ pin["comp"].T
    Zo = _std(Yout, pout["mu"], pout["sd"]) @ pout["comp"].T
    d = Zi.shape[1]
    W = np.linalg.solve(Zi.T @ Zi + lam * np.eye(d), Zi.T @ Zo)
    return dict(pin=pin, pout=pout, W=W)


def ridge_err(m, Xin, Yout):
    Zi = _std(Xin, m["pin"]["mu"], m["pin"]["sd"]) @ m["pin"]["comp"].T
    Zo = _std(Yout, m["pout"]["mu"], m["pout"]["sd"]) @ m["pout"]["comp"].T
    return np.sqrt(((Zo - Zi @ m["W"]) ** 2).mean(1))


def auc(neg, pos):
    allv = np.concatenate([neg, pos]); _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0; r = avg[inv]; n1 = len(pos)
    return (r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1)


def boot(neg, pos, B=1000):
    rng = np.random.default_rng(1); v = np.empty(B)
    for b in range(B):
        v[b] = auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))])
    return np.percentile(v, [2.5, 97.5])


DETS = ["accel_pca", "mic_pca", "concat_pca", "xmodal_a2m", "xmodal_m2a", "xmodal_sym", "xmodal_shuffle"]


def fit_all(An, Mn, k=32):
    return dict(accel=pca_fit(An, k), mic=pca_fit(Mn, k),
               concat=pca_fit(np.hstack([An, Mn]), k),
               a2m=ridge_fit(An, Mn), m2a=ridge_fit(Mn, An))


def score_all(models, A, M):
    rng = np.random.default_rng(7)
    Mperm = M[rng.permutation(len(M))]
    a2m = ridge_err(models["a2m"], A, M)
    m2a = ridge_err(models["m2a"], M, A)
    return {
        "accel_pca": pca_err(models["accel"], A),
        "mic_pca": pca_err(models["mic"], M),
        "concat_pca": pca_err(models["concat"], np.hstack([A, M])),
        "xmodal_a2m": a2m,
        "xmodal_m2a": m2a,
        "xmodal_sym": 0.5 * (a2m + m2a),
        "xmodal_shuffle": ridge_err(models["a2m"], A, Mperm),
    }


def main():
    C = get_cache()
    An, Mn, Sn = C["normal"]; Af, Mf, Sf = C["fault"]
    print(f"normal windows={len(An)} fault windows={len(Af)}; "
          f"speed normal {Sn.min():.0f}-{Sn.max():.0f} Hz, fault {Sf.min():.0f}-{Sf.max():.0f} Hz")

    # ---- matched speed: train on 60% random normal, test rest normal vs fault ----
    print("\n### MATCHED-SPEED (train 60% normal) ###")
    rng = np.random.default_rng(0); idx = rng.permutation(len(An)); cut = int(0.6 * len(An))
    tr, te = idx[:cut], idx[cut:]
    models = fit_all(An[tr], Mn[tr])
    sn = score_all(models, An[te], Mn[te]); sf = score_all(models, Af, Mf)
    for d in DETS:
        a = auc(sn[d], sf[d]); lo, hi = boot(sn[d], sf[d])
        print(f"[{d:15s}] AUC={a:.3f} [{lo:.3f},{hi:.3f}]")

    # ---- confound-free cross-speed: train band A, test normal+fault both on band B!=A ----
    bands = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]
    def binn(arr, s): return [(arr[(s >= lo) & (s < hi)]) for lo, hi in bands]
    Anb, Mnb = binn(An, Sn), binn(Mn, Sn)
    Afb, Mfb = binn(Af, Sf), binn(Mf, Sf)
    print("\n### CONFOUND-FREE CROSS-SPEED (off-diagonal mean AUC) ###")
    agg = {d: [] for d in DETS}
    for ai in range(len(bands)):
        if len(Anb[ai]) < 40:
            continue
        models = fit_all(Anb[ai], Mnb[ai], k=24)
        for bi in range(len(bands)):
            if ai == bi or len(Anb[bi]) < 8 or len(Afb[bi]) < 8:
                continue
            sn = score_all(models, Anb[bi], Mnb[bi]); sf = score_all(models, Afb[bi], Mfb[bi])
            for d in DETS:
                agg[d].append(auc(sn[d], sf[d]))
    for d in DETS:
        v = np.array(agg[d])
        print(f"[{d:15s}] mean={v.mean():.3f}  n={len(v)}  (min {v.min():.2f} max {v.max():.2f})")


if __name__ == "__main__":
    main()
