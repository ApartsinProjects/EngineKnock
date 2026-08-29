"""EXP-003c: confound-free cross-speed test. Train normal-only PCA on speed band A; test
normal AND fault BOTH restricted to a disjoint band B. Now the two test sets share the same
(unseen) speed distribution, so AUC(normal_B vs fault_B) measures fault-detection-at-unseen-
speed with NO speed-distance confound (the flaw in EXP-003b).

Thesis: order-domain features are speed-invariant, so a band-A-trained model still scores
normal_B low and fault_B high -> AUC holds. TIME features move with RPM -> AUC collapses.

Reuses cache_ch1_r16_s256.npz from EXP-003b. Reports per (A,B) AUC + a headline: for each
fixed train band, mean AUC over UNSEEN test bands, ORDER vs TIME.
"""
import os, numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
MODES = ["time", "order", "order_shuffled"]


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:k])


def recon_err(m, X):
    Xs = (X - m["mu"]) / m["sd"]
    return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def auc(neg, pos):
    allv = np.concatenate([neg, pos])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]; n1 = len(pos)
    return (ranks[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1)


def boot(neg, pos, B=1000):
    rng = np.random.default_rng(1); v = np.empty(B)
    for b in range(B):
        v[b] = auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))])
    return np.percentile(v, [2.5, 97.5])


def main():
    cache = np.load(os.path.join(OUT, "cache_ch1_r16_s256.npz"), allow_pickle=True)["obj"].item()
    bands = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]
    K = 24
    # pre-bin windows by speed per mode
    binned = {m: {"normal": [], "fault": []} for m in MODES}
    for m in MODES:
        for lab in ("normal", "fault"):
            X, s = cache[lab][m]
            binned[m][lab] = [X[(s >= lo) & (s < hi)] for (lo, hi) in bands]

    print("bands (Hz):", bands)
    for m in MODES:
        print(f"\n########## MODE = {m} ##########")
        print("rows=train band, cols=test band; cell=AUC(normal_B vs fault_B). diag=matched(control).")
        header = "train\\test " + " ".join(f"{lo}-{hi:>2}" for lo, hi in bands)
        print(header)
        offdiag = []
        for ai, A in enumerate(bands):
            Xtr = binned[m]["normal"][ai]
            row = [f"{A[0]}-{A[1]:>2} "]
            if len(Xtr) < K + 3:
                print(row[0] + " (train too small: %d)" % len(Xtr)); continue
            model = pca_fit(Xtr, k=K)
            for bi, B in enumerate(bands):
                nn = binned[m]["normal"][bi]
                ff = binned[m]["fault"][bi]
                if ai == bi:  # matched-speed control: held-out normal from same band
                    if len(nn) < 10:
                        row.append("  .  "); continue
                    half = len(nn) // 2
                    m2 = pca_fit(nn[:half], k=min(K, half - 2)) if half > K + 2 else model
                    a = auc(recon_err(m2, nn[half:]), recon_err(m2, ff)) if len(ff) else np.nan
                else:
                    if len(nn) < 8 or len(ff) < 8:
                        row.append("  .  "); continue
                    a = auc(recon_err(model, nn), recon_err(model, ff))
                    offdiag.append(a)
                row.append(f"{a:5.2f}")
            print(" ".join(row))
        if offdiag:
            od = np.array(offdiag)
            print(f"  -> off-diagonal (cross-speed) mean AUC = {od.mean():.3f}  "
                  f"(min {od.min():.3f}, max {od.max():.3f}, n={len(od)})")

    # headline: fixed train band = (20,28) low-mid, mean over unseen test bands, per mode
    print("\n===== HEADLINE: train on ONE band (20-28 Hz), test all OTHER bands =====")
    ai = 1
    for m in MODES:
        Xtr = binned[m]["normal"][ai]; model = pca_fit(Xtr, k=K)
        accs = []
        for bi, B in enumerate(bands):
            if bi == ai:
                continue
            nn, ff = binned[m]["normal"][bi], binned[m]["fault"][bi]
            if len(nn) < 8 or len(ff) < 8:
                continue
            accs.append(auc(recon_err(model, nn), recon_err(model, ff)))
        accs = np.array(accs)
        print(f"[{m:16s}] mean cross-speed AUC = {accs.mean():.3f}  (per-band: "
              + ", ".join(f"{a:.2f}" for a in accs) + ")")


if __name__ == "__main__":
    main()
