"""Driver: run the ResearchPivot cycle-2 two-frame (HZ+ORD) fusion experiment on RunPod, cache-hitting
feats.npz from the volume. Full faults (pod has RAM). Result lands in results/twoframe.json."""
import os, traceback
os.environ["FAULT_SUB"] = "0"          # ALL faults (macro-AUC over 5 classes)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("==== twoframe (multi-frame normality, HZ+ORD max-fusion) ====", flush=True)
try:
    import exp_twoframe
    exp_twoframe.main()
    print("==== twoframe OK ====", flush=True)
except Exception:
    print("==== twoframe FAILED ====\n" + traceback.format_exc(), flush=True)
print("[driver] === ALL DONE ===", flush=True)
