"""EXP-003b: SPEED-SHIFT generalization -- the regime where order-sync should win
structurally. Train the normal-only PCA detector on a NARROW rotation-speed band; test on
held-out normal at OTHER speeds + faults at all speeds. Order-domain features are
speed-invariant by construction; time-frequency bins move with RPM, so TIME should degrade
while ORDER holds.

Builds a feature cache (parse each file ONCE, compute all modes + per-window speed), then
runs fast speed-split evals with bootstrap CIs.

Modes: time, order, order_shuffled. Metric: ROC-AUC(normal-test vs fault) + 1000x bootstrap.
"""
import io, os, zipfile
import numpy as np

FS = 50_000
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
MODES = ["time", "order", "order_shuffled"]


def tacho_rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12)
    a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def win_features(sig, rises, revs, sper, mode):
    if len(rises) < revs + 1:
        return
    pulses = rises.copy()
    if mode == "order_shuffled":
        g = np.diff(pulses); g = RNG.permutation(g)
        pulses = np.concatenate([[pulses[0]], pulses[0] + np.cumsum(g)])
    L = revs * sper
    nb = (len(pulses) - 1) // revs
    for b in range(nb):
        p0, p1 = pulses[b * revs], pulses[(b + 1) * revs]
        if p1 <= p0 or p1 > len(sig):
            continue
        if mode.startswith("order"):
            loc = pulses[b * revs:(b + 1) * revs + 1]
            rev = np.interp(np.arange(p0, p1), loc, np.arange(len(loc)))
            grid = np.linspace(rev[0], rev[-1], L, endpoint=False)
            w = np.interp(grid, rev, sig[p0:p1])
        else:
            seg = sig[p0:p1].astype(float)
            w = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
        w = w - w.mean()
        yield np.log1p(np.abs(np.fft.rfft(w * np.hanning(len(w)))))


def build_cache(zip_path, channel, revs, sper, label):
    feats = {m: [] for m in MODES}
    spd = {m: [] for m in MODES}
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for n in names:
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")
            rises = tacho_rises(arr[:, 0])
            frot = FS / np.median(np.diff(rises)) if len(rises) > 2 else np.nan
            for m in MODES:
                for f in win_features(arr[:, channel], rises, revs, sper, m):
                    feats[m].append(f); spd[m].append(frot)
    return {m: (np.asarray(feats[m]), np.asarray(spd[m])) for m in MODES}, label


def get_caches(channel, revs, sper):
    tag = f"ch{channel}_r{revs}_s{sper}"
    p = os.path.join(OUT, f"cache_{tag}.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return d["obj"].item()
    obj = {}
    for zp, lab in [("normal.zip", "normal"),
                    ("horizontal-misalignment.zip", "fault")]:
        c, _ = build_cache(os.path.join(DATA, zp), channel, revs, sper, lab)
        obj[lab] = c
    np.savez_compressed(p, obj=np.array(obj, dtype=object))
    return obj


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xs = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:k])


def recon_err(m, X):
    Xs = (X - m["mu"]) / m["sd"]
    proj = Xs @ m["comp"].T
    return np.sqrt(((Xs - proj @ m["comp"]) ** 2).mean(1))


def auc(neg, pos):
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    n1 = len(pos)
    r1 = ranks[len(neg):].sum()
    return (r1 - n1 * (n1 + 1) / 2) / (len(neg) * n1)


def boot_auc(neg, pos, B=1000):
    rng = np.random.default_rng(1)
    vals = np.empty(B)
    for b in range(B):
        nn = neg[rng.integers(0, len(neg), len(neg))]
        pp = pos[rng.integers(0, len(pos), len(pos))]
        vals[b] = auc(nn, pp)
    return np.percentile(vals, [2.5, 97.5])


def eval_split(caches, band, k=32):
    """band=(lo,hi) Hz training speed window for NORMAL. Test normal = out-of-band normal."""
    print(f"\n=== train-normal speed band {band} Hz ===")
    for m in MODES:
        Xn, sn = caches["normal"][m]
        Xf, sf = caches["fault"][m]
        in_band = (sn >= band[0]) & (sn < band[1])
        Xtr = Xn[in_band]
        Xte_n = Xn[~in_band]           # normal at UNSEEN speeds
        if len(Xtr) < k + 5 or len(Xte_n) < 5:
            print(f"[{m:16s}] insufficient windows (train={len(Xtr)}, test_n={len(Xte_n)})"); continue
        model = pca_fit(Xtr, k=k)
        s_n = recon_err(model, Xte_n)
        s_f = recon_err(model, Xf)
        a = auc(s_n, s_f)
        lo, hi = boot_auc(s_n, s_f)
        print(f"[{m:16s}] train={len(Xtr):4d} test_n(oob)={len(Xte_n):4d} fault={len(Xf):4d} "
              f"| AUC={a:.3f} [{lo:.3f},{hi:.3f}]")


def main():
    caches = get_caches(channel=1, revs=16, sper=256)
    # reference: train on ALL normal (matched speed) -- from EXP-002 we expect ~0.86/0.87
    print("### reference: train on ALL normal (matched speed) ###")
    for m in MODES:
        Xn, sn = caches["normal"][m]; Xf, sf = caches["fault"][m]
        idx = RNG.permutation(len(Xn)); Xn2 = Xn[idx]
        cut = int(0.6 * len(Xn2))
        model = pca_fit(Xn2[:cut]); s_n = recon_err(model, Xn2[cut:]); s_f = recon_err(model, Xf)
        a = auc(s_n, s_f); lo, hi = boot_auc(s_n, s_f)
        print(f"[{m:16s}] AUC={a:.3f} [{lo:.3f},{hi:.3f}]")
    # speed-shift: train only on a narrow LOW band, test across all other speeds
    for band in [(10, 25), (25, 40), (40, 65)]:
        eval_split(caches, band)
    # report the speed distribution for context
    _, sn = caches["normal"]["order"]; _, sf = caches["fault"]["order"]
    print(f"\nnormal speed range {sn.min():.1f}-{sn.max():.1f} Hz; "
          f"fault speed range {sf.min():.1f}-{sf.max():.1f} Hz")


if __name__ == "__main__":
    main()
