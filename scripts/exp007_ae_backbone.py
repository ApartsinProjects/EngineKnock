"""EXP-007: is the speed-conditioning win backbone-agnostic? Compare {PCA, AE} x
{unconditional, conditional} on the POOLED all-fault MaFaulDa set (accel col1), matched-speed
(3 seeds) and cross-speed (LOBO). If conditioning helps under BOTH backbones, the effect is
not a PCA artifact. Memory-light: one fault at a time, pooled via accumulated scores. CPU torch.
"""
import os, json, numpy as np
import torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]


def load(stem):
    d = np.load(os.path.join(OUT, f"mf_{stem}.npz"))
    return d["T"].astype(np.float32), d["S"].astype(np.float32)


def knn(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), dtype=np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        nn_ = c[np.argsort(np.abs(ss[c] - q))[:k]]; out[i] = XX[nn_].mean(0)
    return out


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


# ---- backbones: fit on normal-train X, return scorer f(X)->recon error ----
def pca_backbone(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:k]
    def score(X):
        Xs = (X - mu) / sd; return np.sqrt(((Xs - (Xs @ comp.T) @ comp) ** 2).mean(1))
    return score


class AE(nn.Module):
    def __init__(self, d, lat=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, lat))
        self.dec = nn.Sequential(nn.Linear(lat, 256), nn.ReLU(), nn.Linear(256, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def ae_backbone(Xtr, lat=32, epochs=150):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xs = torch.tensor((Xtr - mu) / sd)
    net = AE(Xtr.shape[1], lat); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); out = net(Xs); loss = lossf(out, Xs); loss.backward(); opt.step()
    net.eval()
    def score(X):
        with torch.no_grad():
            xs = torch.tensor((X - mu) / sd); e = ((net(xs) - xs) ** 2).mean(1)
        return e.numpy()
    return score


BACKBONES = {"PCA": pca_backbone, "AE": ae_backbone}


def run():
    Tn, Sn = load("normal")
    faults = [(s,) + load(s) for s in FAULTS if os.path.exists(os.path.join(OUT, f"mf_{s}.npz"))]
    res = {}
    for bb_name, bb in BACKBONES.items():
        for cond in (False, True):
            key = f"{bb_name}-{'cond' if cond else 'uncond'}"
            # matched 3 seeds
            m_aucs = []
            for seed in range(3):
                rng = np.random.default_rng(seed); idx = rng.permutation(len(Tn)); cut = int(0.6 * len(Tn))
                tr, te = idx[:cut], idx[cut:]
                if cond:
                    base = knn(Tn[tr], Sn[tr], Sn[tr]); scorer = bb(Tn[tr] - base)
                    neg = scorer(Tn[te] - knn(Tn[tr], Sn[tr], Sn[te]))
                    pos = np.concatenate([scorer(Tf - knn(Tn[tr], Sn[tr], Sf)) for _, Tf, Sf in faults])
                else:
                    scorer = bb(Tn[tr]); neg = scorer(Tn[te])
                    pos = np.concatenate([scorer(Tf) for _, Tf, Sf in faults])
                m_aucs.append(auc(neg, pos))
            # LOBO
            l_aucs = []
            for lo, hi in BANDS:
                ten = (Sn >= lo) & (Sn < hi); trn = ~ten
                if ten.sum() < 8 or trn.sum() < 40:
                    continue
                if cond:
                    scorer = bb(Tn[trn] - knn(Tn[trn], Sn[trn], Sn[trn]))
                    neg = scorer(Tn[ten] - knn(Tn[trn], Sn[trn], Sn[ten]))
                    pos = []
                    for _, Tf, Sf in faults:
                        tef = (Sf >= lo) & (Sf < hi)
                        if tef.sum() >= 8:
                            pos.append(scorer(Tf[tef] - knn(Tn[trn], Sn[trn], Sf[tef])))
                else:
                    scorer = bb(Tn[trn]); neg = scorer(Tn[ten])
                    pos = [scorer(Tf[(Sf >= lo) & (Sf < hi)]) for _, Tf, Sf in faults
                           if ((Sf >= lo) & (Sf < hi)).sum() >= 8]
                if pos:
                    l_aucs.append(auc(neg, np.concatenate(pos)))
            res[key] = dict(matched=float(np.mean(m_aucs)), matched_std=float(np.std(m_aucs)),
                            lobo=float(np.mean(l_aucs)))
            print(f"[{key:12s}] matched={res[key]['matched']:.3f}±{res[key]['matched_std']:.2f}  "
                  f"cross-speed(LOBO)={res[key]['lobo']:.3f}")
    json.dump(res, open(os.path.join(OUT, "exp007_ae_results.json"), "w"), indent=2)
    print("\nConditioning effect (cond - uncond):")
    for bb in BACKBONES:
        dm = res[f"{bb}-cond"]["matched"] - res[f"{bb}-uncond"]["matched"]
        dl = res[f"{bb}-cond"]["lobo"] - res[f"{bb}-uncond"]["lobo"]
        print(f"  {bb}: matched {dm:+.3f}, cross-speed {dl:+.3f}")


if __name__ == "__main__":
    run()
