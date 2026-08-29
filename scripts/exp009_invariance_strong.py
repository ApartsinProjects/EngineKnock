"""EXP-009: does the invariance (order-tracking) approach win when given a STRONG deep
backbone? Rebuts 'your invariance baseline is just PCA'. Full {PCA, AE} x {time-uncond,
order-invariance, speed-conditioned} on pooled MaFaulDa (accel col1). Order-tracking + a deep
AE is our reproducible representative of the OASSL / SLAD / CMAE invariance class (OASSL itself
is paywalled with no public code). matched (3 seeds) + LOBO. Reuses mf_*.npz (T=time, O=order,
S=speed). Memory-light score accumulation.
"""
import os, json, numpy as np
import torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]


def load(stem):
    d = np.load(os.path.join(OUT, f"mf_{stem}.npz"))
    return d["T"].astype(np.float32), d["O"].astype(np.float32), d["S"].astype(np.float32)


def knn(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c] - q))[:k]]].mean(0)
    return out


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def pca_backbone(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:k]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


class AE(nn.Module):
    def __init__(s, d, lat=32):
        super().__init__(); s.e = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, lat))
        s.d = nn.Sequential(nn.Linear(lat, 256), nn.ReLU(), nn.Linear(256, d))
    def forward(s, x): return s.d(s.e(x))


def ae_backbone(Xtr, lat=32, epochs=150):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    X = torch.tensor((Xtr - mu) / sd); net = AE(Xtr.shape[1], lat)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5); L = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); l = L(net(X), X); l.backward(); opt.step()
    net.eval()
    def score(A):
        with torch.no_grad():
            xs = torch.tensor((A - mu) / sd); return ((net(xs) - xs) ** 2).mean(1).numpy()
    return score


BACKBONES = {"PCA": pca_backbone, "AE": ae_backbone}
TREAT = ["time_uncond", "order_invariance", "speed_cond"]


def main():
    Tn, On, Sn = load("normal")
    faults = [(load(s)) for s in FAULTS]
    res = {}
    for bb_name, bb in BACKBONES.items():
        for treat in TREAT:
            # matched, 3 seeds
            m = []
            for seed in range(3):
                rng = np.random.default_rng(seed); idx = rng.permutation(len(Tn)); cut = int(0.6 * len(Tn))
                tr, te = idx[:cut], idx[cut:]
                if treat == "time_uncond":
                    sc = bb(Tn[tr]); neg = sc(Tn[te]); pos = np.concatenate([sc(Tf) for Tf, Of, Sf in faults])
                elif treat == "order_invariance":
                    sc = bb(On[tr]); neg = sc(On[te]); pos = np.concatenate([sc(Of) for Tf, Of, Sf in faults])
                else:
                    sc = bb(Tn[tr] - knn(Tn[tr], Sn[tr], Sn[tr]))
                    neg = sc(Tn[te] - knn(Tn[tr], Sn[tr], Sn[te]))
                    pos = np.concatenate([sc(Tf - knn(Tn[tr], Sn[tr], Sf)) for Tf, Of, Sf in faults])
                m.append(auc(neg, pos))
            # LOBO
            l = []
            for lo, hi in BANDS:
                ten = (Sn >= lo) & (Sn < hi); trn = ~ten
                if ten.sum() < 8 or trn.sum() < 40:
                    continue
                if treat == "time_uncond":
                    sc = bb(Tn[trn]); neg = sc(Tn[ten])
                    pos = [sc(Tf[(Sf >= lo) & (Sf < hi)]) for Tf, Of, Sf in faults if ((Sf >= lo) & (Sf < hi)).sum() >= 8]
                elif treat == "order_invariance":
                    sc = bb(On[trn]); neg = sc(On[ten])
                    pos = [sc(Of[(Sf >= lo) & (Sf < hi)]) for Tf, Of, Sf in faults if ((Sf >= lo) & (Sf < hi)).sum() >= 8]
                else:
                    sc = bb(Tn[trn] - knn(Tn[trn], Sn[trn], Sn[trn]))
                    neg = sc(Tn[ten] - knn(Tn[trn], Sn[trn], Sn[ten]))
                    pos = [sc(Tf[(Sf >= lo) & (Sf < hi)] - knn(Tn[trn], Sn[trn], Sf[(Sf >= lo) & (Sf < hi)]))
                           for Tf, Of, Sf in faults if ((Sf >= lo) & (Sf < hi)).sum() >= 8]
                if pos:
                    l.append(auc(neg, np.concatenate(pos)))
            res[f"{bb_name}-{treat}"] = dict(matched=float(np.mean(m)), matched_std=float(np.std(m)), lobo=float(np.mean(l)))
            print(f"[{bb_name}-{treat:16s}] matched={res[f'{bb_name}-{treat}']['matched']:.3f}  cross-speed={res[f'{bb_name}-{treat}']['lobo']:.3f}")
    json.dump(res, open(os.path.join(OUT, "exp009_results.json"), "w"), indent=2)
    print("\nInvariance vs conditioning at each backbone (cross-speed LOBO):")
    for bb in BACKBONES:
        print(f"  {bb}: uncond {res[f'{bb}-time_uncond']['lobo']:.3f}, invariance {res[f'{bb}-order_invariance']['lobo']:.3f}, conditioning {res[f'{bb}-speed_cond']['lobo']:.3f}")


if __name__ == "__main__":
    main()
