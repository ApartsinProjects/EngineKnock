"""Objection-(iv) neutralizer: a speed-CONDITIONED autoencoder baseline, the deep counterpart to the
linear 'adjust'. Trained on healthy REV features with the (normalized) speed appended to the input and
to the bottleneck; scored by reconstruction error. Reports leakage-free LOSO AUC against ignore / adjust
(kNN local-mean + PCA) so the paper can state whether the lightweight linear treatment matches or beats a
conditioned deep detector. CPU torch (no CUDA). Loads the shared feature cache (feats.npz)."""
import os, sys, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_strongaccept_v2 as J


def knn_mean(Xtr, st, sq, k=15):
    return J.knn_mean(Xtr, st, sq, k)


def pca(Xtr, k=32):
    return J.pca(Xtr, k)


def condae_scorer(Xtr, Str, epochs=60, bott=16, hid=128, seed=0):
    """Train a speed-conditioned AE on healthy Xtr; return a scorer f(Xq,Sq)->recon error."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    smu = Str.mean(); ssd = Str.std() + 1e-8
    D = Xtr.shape[1]
    def prep(X, S):
        xn = (X - mu) / sd; sn = ((S - smu) / ssd)[:, None]
        return np.concatenate([xn, sn], 1).astype(np.float32)
    enc = nn.Sequential(nn.Linear(D + 1, hid), nn.ReLU(), nn.Linear(hid, bott))
    dec = nn.Sequential(nn.Linear(bott + 1, hid), nn.ReLU(), nn.Linear(hid, D))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
    Xin = torch.tensor(prep(Xtr, Str)); Xtgt = torch.tensor(((Xtr - mu) / sd).astype(np.float32))
    sn_tr = Xin[:, -1:]
    n = len(Xin); bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            z = enc(Xin[idx]); rec = dec(torch.cat([z, sn_tr[idx]], 1))
            loss = ((rec - Xtgt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); dec.eval()
    def f(Xq, Sq, ch=4000):
        out = np.empty(len(Xq), np.float32)
        with torch.no_grad():
            for i in range(0, len(Xq), ch):
                xi = torch.tensor(prep(Xq[i:i+ch], Sq[i:i+ch]))
                z = enc(xi); rec = dec(torch.cat([z, xi[:, -1:]], 1))
                tgt = torch.tensor(((Xq[i:i+ch] - mu) / sd).astype(np.float32))
                out[i:i+ch] = ((rec - tgt) ** 2).mean(1).numpy()
        return out
    return f


def main():
    from sklearn.metrics import roc_auc_score
    D = J.get_features()
    Xn, Sn, Rn = D["Nh_REV"], D["Nh_S"], D["Nh_R"]
    Xf, Sf = D["F_REV"], D["F_S"]
    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    recs = np.unique(Rn); acc = {m: [] for m in ("ignore", "adjust", "condae")}
    for rid in recs:
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        fi = pca(Xn[tr]); acc["ignore"].append(auc(fi(Xn[te]), fi(Xf)))
        fa = pca(Xn[tr] - knn_mean(Xn[tr], Sn[tr], Sn[tr]))
        acc["adjust"].append(auc(fa(Xn[te] - knn_mean(Xn[tr], Sn[tr], Sn[te])), fa(Xf - knn_mean(Xn[tr], Sn[tr], Sf))))
        fc = condae_scorer(Xn[tr], Sn[tr])
        acc["condae"].append(auc(fc(Xn[te], Sn[te]), fc(Xf, Sf)))
        print(f"[condae] rec {rid}: ignore {acc['ignore'][-1]:.3f} adjust {acc['adjust'][-1]:.3f} condae {acc['condae'][-1]:.3f}", flush=True)
    res = {m: {"mean": round(float(np.mean(v)), 3), "n": len(v)} for m, v in acc.items()}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/condae.json", "w"), indent=2)
    print("[condae] SUMMARY", json.dumps(res), flush=True)
    print("[condae] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
