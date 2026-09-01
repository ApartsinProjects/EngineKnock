"""Real running-engine audio anomaly detection (Kaggle car-diagnostics), the user's exact pipeline:
train a reconstructor on NORMAL engine sound only, score by reconstruction residual, alarm over a
healthy-set threshold. Frequency-domain features (average log power spectrum). Two backbones: PCA and
a dense autoencoder. Evaluated on the IDLE state (near-fixed operating condition, so this validates the
BASE detector, not the speed-conditioning contribution which lives on MaFaulDa).
Reports per-fault ROC-AUC (normal-test vs each fault), pooled AUC, and a healthy-threshold FPR/TPR.
Also runs the STARTUP state (normal vs bad_ignition / dead_battery) as a secondary combustion-relevant cut."""
import os, io, json, zipfile, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score

ZIP = "data/kaggle/car_diagnostics.zip"
NFFT = 2048; HOP = 1024


def wav_feature(b):
    sr, x = wavfile.read(io.BytesIO(b))
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    x = x - x.mean()
    if np.abs(x).max() > 0:
        x = x / (np.abs(x).max() + 1e-9)
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    win = np.hanning(NFFT)
    ps = []
    for i in range(0, len(x) - NFFT + 1, HOP):
        seg = x[i:i + NFFT] * win
        ps.append(np.abs(np.fft.rfft(seg)) ** 2)
    if not ps:
        return None
    avg = np.mean(ps, 0)
    return np.log1p(avg).astype(np.float32)


def load_state(state):
    z = zipfile.ZipFile(ZIP)
    cls = {}
    for n in z.namelist():
        if f"{state}/" not in n or not n.lower().endswith(".wav"):
            continue
        c = n.split("/")[-2]
        f = wav_feature(z.read(n))
        if f is not None:
            cls.setdefault(c, []).append(f)
    return {c: np.vstack(v) for c, v in cls.items()}


def pca_scorer(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
    comp = Vt[:min(k, max(1, Xtr.shape[0] - 2))]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def ae_scorer(Xtr, epochs=120, bott=16, hid=128, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8; D = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Linear(hid, bott), nn.ReLU(),
                        nn.Linear(bott, hid), nn.ReLU(), nn.Linear(hid, D))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    X = torch.tensor(((Xtr - mu) / sd).astype(np.float32)); n = len(X); bs = 64
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; rec = net(X[idx]); loss = ((rec - X[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def f(Y):
        with torch.no_grad():
            yt = torch.tensor(((Y - mu) / sd).astype(np.float32)); rec = net(yt)
            return ((rec - yt) ** 2).mean(1).numpy()
    return f


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def evaluate(state, normal_key, seed=0):
    data = load_state(state)
    if normal_key not in data:
        print(f"[audio] {state}: no normal key {normal_key}; have {list(data)}"); return None
    Xn = data[normal_key]; faults = {c: v for c, v in data.items() if c != normal_key}
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Xn))
    cut = int(0.7 * len(Xn)); tr, te = idx[:cut], idx[cut:]
    out = {"state": state, "normal": normal_key, "n_normal": len(Xn),
           "n_faults": {c: len(v) for c, v in faults.items()}, "dim": int(Xn.shape[1])}
    for name, mk in [("pca", pca_scorer), ("ae", ae_scorer)]:
        f = mk(Xn[tr])
        sn = f(Xn[te]); thr = float(np.percentile(f(Xn[tr]), 99))
        per = {c: round(auc(sn, f(v)), 3) for c, v in faults.items()}
        allf = np.vstack(list(faults.values()))
        pooled = round(auc(sn, f(allf)), 3)
        tpr = {c: round(float((f(v) > thr).mean()), 3) for c, v in faults.items()}
        out[name] = {"pooled_auc": pooled, "per_fault_auc": per,
                     "healthy_fpr": round(float((sn > thr).mean()), 3), "per_fault_tpr@99": tpr}
        print(f"[audio] {state}/{name}: pooled AUC {pooled} | per-fault {per} | healthy FPR {out[name]['healthy_fpr']}", flush=True)
    return out


def main():
    res = {}
    res["idle"] = evaluate("idle state", "normal_engine_idle")
    res["startup"] = evaluate("startup state", "normal_engine_startup")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/engine_audio.json", "w"), indent=2)
    print("[audio] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
