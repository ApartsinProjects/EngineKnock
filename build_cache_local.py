"""Build the MaFaulDa feature cache LOCALLY from the raw zips already in data/, one stem at a time
(bounds peak memory to a single fault), then assemble the exact npz layout runpod_strongaccept_v2 and
runpod_recipe expect and upload it to the RunPod volume so every future pod job cache-HITs in minutes.
Reuses extract() from runpod_strongaccept_v2 so features are byte-identical to the pod pipeline."""
import os, sys, json, gc, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_strongaccept_v2 as J

PARTS = "results/cache_parts"; os.makedirs(PARTS, exist_ok=True)
STEMS = ["normal"] + J.FAULTS


def build_parts():
    for stem in STEMS:
        out = f"{PARTS}/{stem}.npz"
        if os.path.exists(out):
            print(f"[cache] part exists {stem}", flush=True); continue
        print(f"[cache] extracting {stem} ...", flush=True)
        H, R_, O, S, REC = J.extract(stem)
        np.savez_compressed(out, HZ=H, REV=R_, ORD=O, S=S, REC=REC)
        print(f"[cache]   {stem}: {len(S)} windows, dims {H.shape[1]}/{R_.shape[1]}/{O.shape[1]}", flush=True)
        del H, R_, O, S, REC; gc.collect()


def assemble():
    n = np.load(f"{PARTS}/normal.npz")
    out = dict(Nh_HZ=n["HZ"], Nh_REV=n["REV"], Nh_ORD=n["ORD"], Nh_S=n["S"], Nh_R=n["REC"])
    fh, fr, fo, fs, ff = [], [], [], [], []
    for i, stem in enumerate(J.FAULTS):
        d = np.load(f"{PARTS}/{stem}.npz")
        fh.append(d["HZ"]); fr.append(d["REV"]); fo.append(d["ORD"]); fs.append(d["S"])
        ff.append(np.full(len(d["S"]), i))
    out.update(F_HZ=np.vstack(fh), F_REV=np.vstack(fr), F_ORD=np.vstack(fo),
               F_S=np.concatenate(fs), F_FAULT=np.concatenate(ff))
    np.savez_compressed("feats.npz", **out)
    mb = os.path.getsize("feats.npz")/1e6
    print(f"[cache] wrote feats.npz ({mb:.0f} MB); healthy={len(out['Nh_S'])} fault={len(out['F_S'])}", flush=True)
    return mb


def upload():
    try:
        cli, vol = J.s3()
        cli.upload_file("feats.npz", vol, J.CACHE_KEY)
        print(f"[cache] uploaded to volume {vol}/{J.CACHE_KEY}", flush=True)
    except Exception as e:
        print(f"[cache] upload FAILED (will retry from a pod or later): {str(e)[:120]}", flush=True)


def build_chunk(stem, part, nparts):
    """Extract only recordings [part::nparts] of a stem (halves peak memory for huge zips).
    Fault analysis ignores per-recording IDs, so slicing the CSV list is safe for faults."""
    import io, zipfile
    out = f"{PARTS}/{stem}_{part}of{nparts}.npz"
    if os.path.exists(out):
        print(f"[cache] chunk exists {stem} {part}/{nparts}", flush=True); return
    p = f"data/{stem}.zip"
    H, R_, O, S = [], [], [], []
    with zipfile.ZipFile(p) as zf:
        names = sorted(x for x in zf.namelist() if x.lower().endswith(".csv"))
        sel = names[part::nparts]
        print(f"[cache] {stem} chunk {part}/{nparts}: {len(sel)}/{len(names)} recordings", flush=True)
        for n in sel:
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=","); r = J.rises(arr[:, 0])
            for a, b, c in J.feats(arr[:, J.ACC], r):
                H.append(a); R_.append(b); O.append(c); S.append(J.FS/np.median(np.diff(r)))
    np.savez_compressed(out, HZ=np.asarray(H, np.float32), REV=np.asarray(R_, np.float32),
                        ORD=np.asarray(O, np.float32), S=np.asarray(S, np.float32),
                        REC=np.zeros(len(S), int))
    print(f"[cache]   {stem} chunk {part}/{nparts}: {len(S)} windows", flush=True)


def merge_chunks(stem, nparts):
    out = f"{PARTS}/{stem}.npz"
    if os.path.exists(out):
        return
    ps = [np.load(f"{PARTS}/{stem}_{i}of{nparts}.npz") for i in range(nparts)]
    np.savez_compressed(out,
        HZ=np.vstack([d["HZ"] for d in ps]), REV=np.vstack([d["REV"] for d in ps]),
        ORD=np.vstack([d["ORD"] for d in ps]), S=np.concatenate([d["S"] for d in ps]),
        REC=np.concatenate([d["REC"] for d in ps]))
    print(f"[cache] merged {stem} from {nparts} chunks", flush=True)


def build_one(stem):
    out = f"{PARTS}/{stem}.npz"
    if os.path.exists(out):
        print(f"[cache] part exists {stem}", flush=True); return
    print(f"[cache] extracting {stem} ...", flush=True)
    H, R_, O, S, REC = J.extract(stem)
    np.savez_compressed(out, HZ=H, REV=R_, ORD=O, S=S, REC=REC)
    print(f"[cache]   {stem}: {len(S)} windows, dims {H.shape[1]}/{R_.shape[1]}/{O.shape[1]}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        assemble(); upload(); print("[cache] === DONE ===", flush=True)
    elif len(sys.argv) > 3 and sys.argv[1] == "chunk":
        build_chunk(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))  # chunk <stem> <part> <nparts>
    elif len(sys.argv) > 2 and sys.argv[1] == "merge":
        merge_chunks(sys.argv[2], int(sys.argv[3]))                    # merge <stem> <nparts>
    elif len(sys.argv) > 1:
        build_one(sys.argv[1])  # extract ONE stem in an isolated process
    else:
        build_parts(); assemble(); upload(); print("[cache] === DONE ===", flush=True)
