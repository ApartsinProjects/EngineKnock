"""EXP-002/003: unsupervised anomaly detection on MaFaulDa, TIME-spectrum baseline vs
ORDER-synchronous representation, at equal backbone (PCA reconstruction-error detector).

Construct-matched: both feature modes use the SAME revolution-defined windows, the SAME
output dimensionality, and the SAME detector, computed in ONE pass. Score = PCA recon error.
Headline = ROC-AUC separating held-out NORMAL windows from FAULT windows.

Controls (stated in advance):
  * degenerate invariant: ORDER with a constant-speed (uniform) angle map must ~= TIME mode.
  * tacho-shuffle: ORDER with randomly shuffled pulse gaps must drop AUC toward/under TIME.
  * normal-vs-normal null: held-out normal vs a second held-out normal split ~ chance (0.5).

Usage: python exp002_baseline_ordersync.py [--channel 1] [--revs 16] [--sper 256]
"""
import argparse, io, os, zipfile
import numpy as np

FS = 50_000
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


# ----------------------------- io -----------------------------
def read_zip_csvs(path):
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for n in names:
            yield n, np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")


def tacho_rises(tacho):
    x = tacho.astype(float)
    x = (x - x.min()) / (np.ptp(x) + 1e-12)
    above = x > 0.5
    return np.where((~above[:-1]) & (above[1:]))[0]


# ----------------------- feature extraction -----------------------
def windows_by_revs(sig, rises, revs_per_win, sper, mode="order", shuffle=False):
    """Yield fixed-length feature vectors, one per non-overlapping block of `revs_per_win`
    revolutions. mode: 'order' (tacho angle map), 'time' (uniform over the same span),
    'degenerate' (order pipeline but uniform angle map -> should equal time)."""
    if len(rises) < revs_per_win + 1:
        return
    pulses = rises.copy()
    if shuffle:  # tacho-shuffle control: keep #pulses & span, permute the gaps
        gaps = np.diff(pulses)
        gaps = RNG.permutation(gaps)
        pulses = np.concatenate([[pulses[0]], pulses[0] + np.cumsum(gaps)])
    n_blocks = (len(pulses) - 1) // revs_per_win
    L = revs_per_win * sper
    for b in range(n_blocks):
        p0, p1 = pulses[b * revs_per_win], pulses[(b + 1) * revs_per_win]
        if p1 <= p0 or p1 > len(sig):
            continue
        if mode == "order":
            local = pulses[b * revs_per_win:(b + 1) * revs_per_win + 1]
            revidx = np.arange(len(local))
            idx = np.arange(p0, p1)
            rev = np.interp(idx, local, revidx)
            grid = np.linspace(rev[0], rev[-1], L, endpoint=False)
            w = np.interp(grid, rev, sig[p0:p1])
        else:  # 'time' / 'degenerate': uniform resample of the same time span to L points
            seg = sig[p0:p1].astype(float)
            w = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
        w = w - w.mean()
        spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
        yield np.log1p(spec)


def features_from_zip(path, channel, revs, sper, mode, shuffle=False, limit=None):
    feats, speeds = [], []
    for i, (name, arr) in enumerate(read_zip_csvs(path)):
        if limit and i >= limit:
            break
        rises = tacho_rises(arr[:, 0])
        frot = FS / np.median(np.diff(rises)) if len(rises) > 2 else np.nan
        for f in windows_by_revs(arr[:, channel], rises, revs, sper, mode, shuffle):
            feats.append(f); speeds.append(frot)
    return np.array(feats), np.array(speeds)


# ----------------------------- detector -----------------------------
def pca_fit(X, k=32):
    mu = X.mean(0); Xc = X - mu
    # standardize per-dim to avoid a few high-energy bins dominating
    sd = Xc.std(0) + 1e-8; Xs = Xc / sd
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    comp = Vt[:k]
    return dict(mu=mu, sd=sd, comp=comp)


def pca_recon_error(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    proj = Xs @ model["comp"].T
    rec = proj @ model["comp"]
    return np.sqrt(((Xs - rec) ** 2).mean(1))


def auc(neg, pos):
    """AUC = P(score_pos > score_neg). neg=normal-test, pos=fault."""
    from bisect import bisect_left, bisect_right
    s = np.sort(neg); n = len(neg); tot = 0.0
    for v in pos:
        lo = bisect_left(s, v); hi = bisect_right(s, v)
        tot += lo + (hi - lo) * 0.5
    return tot / (n * len(pos))


# ----------------------------- main -----------------------------
def run(channel=1, revs=16, sper=256, k=32):
    normal_zip = os.path.join(DATA, "normal.zip")
    fault_zip = os.path.join(DATA, "horizontal-misalignment.zip")
    def _valid_zip(p):
        try:
            with zipfile.ZipFile(p) as zf:  # constructor validates central directory
                return any(n.lower().endswith(".csv") for n in zf.namelist())
        except Exception:
            return False
    have_fault = _valid_zip(fault_zip)
    results = {}
    for mode, shuffle, tag in [("time", False, "TIME"),
                               ("order", False, "ORDER"),
                               ("degenerate", False, "ORDER-degenerate(invariant)"),
                               ("order", True, "ORDER-shuffled(control)")]:
        Xn, spn = features_from_zip(normal_zip, channel, revs, sper, mode, shuffle)
        # split normal 60/40 into train / test-normal (two halves also give the null)
        idx = RNG.permutation(len(Xn)); Xn = Xn[idx]
        cut = int(0.6 * len(Xn))
        Xtr, Xte_n = Xn[:cut], Xn[cut:]
        half = len(Xte_n) // 2
        Xte_n1, Xte_n2 = Xte_n[:half], Xte_n[half:]
        model = pca_fit(Xtr, k=k)
        s_n1 = pca_recon_error(model, Xte_n1)
        s_n2 = pca_recon_error(model, Xte_n2)
        null_auc = auc(s_n1, s_n2)  # should be ~0.5
        line = f"[{tag:32s}] dim={Xtr.shape[1]} ntrain={len(Xtr)} | null(N-vs-N) AUC={null_auc:.3f}"
        if have_fault:
            Xf, spf = features_from_zip(fault_zip, channel, revs, sper, mode, shuffle)
            s_f = pca_recon_error(model, Xf)
            fault_auc = auc(np.concatenate([s_n1, s_n2]), s_f)
            line += f" | FAULT AUC={fault_auc:.3f} (nfault={len(Xf)})"
            results[tag] = dict(fault_auc=fault_auc, null_auc=null_auc)
        else:
            results[tag] = dict(null_auc=null_auc)
        print(line)
    np.save(os.path.join(OUT, f"exp002_results_ch{channel}_r{revs}_s{sper}.npy"),
            np.array([results], dtype=object))
    print("\nfault data present:", have_fault)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--revs", type=int, default=16)
    ap.add_argument("--sper", type=int, default=256)
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    run(a.channel, a.revs, a.sper, a.k)
