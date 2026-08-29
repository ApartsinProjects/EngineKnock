"""EXP-003d: salvage test. Does a HYBRID (time-spectrum concat order-spectrum) beat TIME
alone in the confound-free cross-speed regime? Mechanism: time holds fixed-Hz resonances,
order holds shaft harmonics; if complementary, the union generalizes better across speed.
Reuses cache_ch1_r16_s256.npz. Honest gate: hybrid must beat time by a clear margin with
non-overlapping bootstrap CIs, else the order channel adds nothing and the thesis stays dead.
"""
import os, numpy as np
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:min(k, X.shape[0] - 2)])


def rerr(m, X):
    Xs = (X - m["mu"]) / m["sd"]
    return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def auc(neg, pos):
    allv = np.concatenate([neg, pos]); u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0; ranks = avg[inv]; n1 = len(pos)
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

    def binm(mode, lab):
        X, s = cache[lab][mode]
        return [X[(s >= lo) & (s < hi)] for lo, hi in bands]

    # standardize each modality by TRAIN stats then concat for hybrid; here just concat raw
    # (pca_fit standardizes per-dim anyway).
    reps = {
        "time": (binm("time", "normal"), binm("time", "fault")),
        "order": (binm("order", "normal"), binm("order", "fault")),
    }
    # hybrid = concat time|order per band
    hn = [np.hstack([reps["time"][0][i], reps["order"][0][i]]) for i in range(len(bands))]
    hf = [np.hstack([reps["time"][1][i], reps["order"][1][i]]) for i in range(len(bands))]
    reps["hybrid"] = (hn, hf)

    print("Confound-free cross-speed: off-diagonal mean AUC (train band A -> test band B!=A)")
    summary = {}
    for name, (Ns, Fs) in reps.items():
        offd = []
        for ai in range(len(bands)):
            if len(Ns[ai]) < K + 3:
                continue
            m = pca_fit(Ns[ai], k=K)
            for bi in range(len(bands)):
                if ai == bi or len(Ns[bi]) < 8 or len(Fs[bi]) < 8:
                    continue
                offd.append(auc(rerr(m, Ns[bi]), rerr(m, Fs[bi])))
        offd = np.array(offd); summary[name] = offd
        print(f"[{name:8s}] mean={offd.mean():.3f}  n={len(offd)}  (min {offd.min():.2f} max {offd.max():.2f})")

    # paired comparison hybrid vs time on the SAME (A,B) cells + bootstrap on the paired diff
    t, h = summary["time"], summary["hybrid"]
    if len(t) == len(h):
        d = h - t
        rng = np.random.default_rng(3)
        bd = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(bd, [2.5, 97.5])
        print(f"\nHYBRID - TIME paired mean diff = {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"({'SIGNIFICANT' if lo > 0 else 'not significant'})")

    print("\nHEADLINE: train (20-28), test other bands")
    ai = 1
    for name, (Ns, Fs) in reps.items():
        m = pca_fit(Ns[ai], k=K); accs = []
        for bi in range(len(bands)):
            if bi == ai or len(Ns[bi]) < 8 or len(Fs[bi]) < 8:
                continue
            accs.append(auc(rerr(m, Ns[bi]), rerr(m, Fs[bi])))
        accs = np.array(accs)
        print(f"[{name:8s}] mean={accs.mean():.3f}  per-band=" + ", ".join(f"{a:.2f}" for a in accs))


if __name__ == "__main__":
    main()
