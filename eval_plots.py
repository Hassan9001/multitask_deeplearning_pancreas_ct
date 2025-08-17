# eval_plots.py
import os, argparse, pickle, json, math
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.special
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def _cm_plot(M, classes, path, title, fmt):
    import matplotlib.pyplot as plt, numpy as np
    plt.figure(figsize=(6,5))
    plt.imshow(M, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    # annotate
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{M[i,j]:{fmt}}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_confusion_simple(labels, probs, out_prefix, class_names=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(labels, int)
    y_pred = np.argmax(probs, axis=1)
    n = probs.shape[1]
    if not class_names or len(class_names) != n:
        class_names = [f"subtype{i}" for i in range(n)]
    # ensure all classes appear on axes
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n))
    # raw
    _cm_plot(cm, class_names, f"{out_prefix}_raw.png", "Confusion (counts)", "d")
    # row-normalized (safe div)
    row = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, np.maximum(row, 1), where=row>0)
    _cm_plot(cmn, class_names, f"{out_prefix}_norm.png", "Confusion (row %)", ".2f")

def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# def load_cls_results(pkl_path):
#     with open(pkl_path, "rb") as f:
#         obj = pickle.load(f)
#     # Make robust to different key names
#     logits = obj.get("logits") or obj.get("y_logits") or obj.get("pred_logits") or obj.get("preds")
#     probs  = obj.get("probs")
#     labels = obj.get("labels") or obj.get("y_true") or obj.get("targets")
#     ids    = obj.get("ids") or obj.get("keys") or obj.get("case_ids")
#     if probs is None and logits is not None:
#         probs = softmax(logits)
#     if probs is None or labels is None:
#         raise RuntimeError("classification_results.pkl missing probs/logits or labels")
#     probs = np.asarray(probs)
#     labels = np.asarray(labels).astype(int)
#     if ids is None:
#         ids = [f"case_{i}" for i in range(len(labels))]
#     return probs, labels, ids
# --- replace load_cls_results with this ---
def load_cls_results(pkl_path, csv_path=None):
    import numpy as np, pickle, pandas as pd, re
    def _softmax(a):
        a = np.asarray(a)
        a = a - np.max(a, axis=-1, keepdims=True)
        e = np.exp(a)
        return e / np.sum(e, axis=-1, keepdims=True)

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # Case 1: dict {logits|probs, labels, ids}
    if isinstance(obj, dict):
        logits = (obj.get("logits") or obj.get("y_logits") or obj.get("pred_logits") or obj.get("scores") or obj.get("preds"))
        probs  = obj.get("probs")
        labels = (obj.get("labels") or obj.get("y_true") or obj.get("targets") or obj.get("label"))
        ids    = (obj.get("ids") or obj.get("keys") or obj.get("case_ids") or obj.get("names"))
        if probs is None and logits is not None:
            probs = _softmax(logits)
        if probs is not None and labels is not None:
            if ids is None: ids = [f"case_{i}" for i in range(len(labels))]
            return np.asarray(probs), np.asarray(labels).astype(int), list(ids)

    # Case 2: list-like
    if isinstance(obj, list) and len(obj):
        first = obj[0]

        # 2a) list of dicts
        if isinstance(first, dict):
            LOGIT_KEYS = ["logits","y_logits","pred_logits","scores","preds"]
            PROB_KEYS  = ["probs","probabilities","y_prob"]
            LABEL_KEYS = ["label","labels","y_true","target","class","gt"]
            ID_KEYS    = ["id","case","case_id","name","key","filename"]
            probs_list, labels_list, ids_list = [], [], []
            for i, d in enumerate(obj):
                # label
                lab = None
                for k in LABEL_KEYS:
                    if k in d: lab = int(d[k]); break
                # id
                cid = None
                for k in ID_KEYS:
                    if k in d: cid = d[k]; break
                if cid is None: cid = f"case_{i}"
                # probs/logits
                p = None
                for k in PROB_KEYS:
                    if k in d: p = np.asarray(d[k]); break
                if p is None:
                    lg = None
                    for k in LOGIT_KEYS:
                        if k in d: lg = np.asarray(d[k]); break
                    if lg is not None: p = _softmax(lg)
                if p is None or lab is None:
                    raise RuntimeError("List-of-dicts PKL missing probs/logits or label.")
                probs_list.append(p); labels_list.append(lab); ids_list.append(cid)
            probs = np.vstack(probs_list) if probs_list and np.ndim(probs_list[0])>0 else np.array(probs_list)
            return probs, np.asarray(labels_list).astype(int), ids_list

        # 2b) list of triples in *any* order (id, label, vec)
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            ids, labels, vecs = [], [], []
            for i, tup in enumerate(obj):
                tup = list(tup)
                # identify parts
                c_id = next((x for x in tup if isinstance(x, str)), None)
                c_lab = next((x for x in tup if isinstance(x, (int, np.integer))), None)
                c_vec = next((x for x in tup if hasattr(x, "__len__") and not isinstance(x, (str, bytes)) and np.ndim(np.asarray(x))>=1), None)
                if c_vec is None or c_lab is None:
                    vecs = None; break
                ids.append(c_id if c_id is not None else f"case_{i}")
                labels.append(int(c_lab))
                vecs.append(np.asarray(c_vec))
            if vecs is not None:
                vecs = np.vstack(vecs) if np.ndim(vecs[0])>0 else np.array(vecs)
                # treat as logits if not in [0,1]
                if (vecs.max() > 1.0) or (vecs.min() < 0.0):
                    probs = _softmax(vecs)
                else:
                    probs = vecs
                return probs, np.asarray(labels).astype(int), ids

        # 2c) list of vectors only -> need CSV for labels (and maybe ids)
        try:
            vecs = np.asarray(obj)
            if vecs.ndim == 2:
                probs = vecs if (0.0 <= vecs.min() <= vecs.max() <= 1.0) else _softmax(vecs)
                if csv_path and os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    # find label, id, probs/logits columns if present
                    lab_col = next((c for c in df.columns if c.lower() in ["label","labels","y_true","target","class","gt"]), None)
                    id_col  = next((c for c in df.columns if c.lower() in ["id","case","case_id","name","filename","key"]), None)
                    if lab_col is None:
                        raise RuntimeError("CSV fallback found but no label column.")
                    labels = df[lab_col].astype(int).to_numpy()
                    if len(labels) != probs.shape[0]:
                        # try to align by trimming to min length
                        n = min(len(labels), probs.shape[0])
                        labels = labels[:n]; probs = probs[:n]
                    ids = df[id_col].astype(str).tolist()[:probs.shape[0]] if id_col else [f"case_{i}" for i in range(probs.shape[0])]
                    return probs, labels, ids
        except Exception:
            pass

    # Final fallback: CSV-only (no usable PKL)
    if csv_path and os.path.exists(csv_path):
        import pandas as pd, numpy as np
        df = pd.read_csv(csv_path)
        lab_col = next((c for c in df.columns if c.lower() in ["label","labels","y_true","target","class","gt"]), None)
        id_col  = next((c for c in df.columns if c.lower() in ["id","case","case_id","name","filename","key"]), None)
        # try to find per-class columns like prob_0, prob_1, ...
        prob_cols = [c for c in df.columns if re.match(r"(prob|p|logit)[_\-]?\d+$", c.lower())]
        if not prob_cols:
            # also accept class0/class1/class2
            prob_cols = [c for c in df.columns if re.match(r"(class|c)[_\-]?\d+$", c.lower())]
        if lab_col is None or not prob_cols:
            raise RuntimeError("CSV fallback present but could not find labels and per-class probabilities/logits.")
        probs = df[prob_cols].to_numpy()
        if probs.max() > 1.0 or probs.min() < 0.0:
            probs = _softmax(probs)
        labels = df[lab_col].astype(int).to_numpy()
        ids = df[id_col].astype(str).tolist() if id_col else [f"case_{i}" for i in range(len(labels))]
        return probs, labels, ids

    raise RuntimeError(f"Unsupported PKL format and no usable CSV at {csv_path}")

def plot_confusion(y_true, y_pred, class_names, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for mat, name in [(cm, "Confusion Matrix"), (cmn, "Normalized Confusion Matrix")]:
        plt.figure()
        plt.imshow(mat, interpolation="nearest")
        plt.title(name)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt = f"{mat[i,j]:.2f}" if mat is cmn else str(mat[i,j])
                plt.text(j, i, txt, ha="center", va="center")
        base = out_png if "confusion" in out_png else out_png.replace(".png","")
        suffix = "norm" if name.startswith("Normalized") else "raw"
        plt.tight_layout()
        plt.savefig(f"{base}_{suffix}.png", dpi=150)
        plt.close()

def plot_roc_pr(y_true, probs, class_names, out_prefix):
    # one-vs-rest
    n_classes = probs.shape[1]
    y_bin = np.eye(n_classes)[y_true]
    # ROC
    plt.figure()
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,c], probs[:,c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (one-vs-rest)")
    plt.legend()
    plt.tight_layout(); plt.savefig(f"{out_prefix}_roc.png", dpi=150); plt.close()
    # PR
    plt.figure()
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:,c], probs[:,c])
        ap = average_precision_score(y_bin[:,c], probs[:,c])
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (one-vs-rest)")
    plt.legend()
    plt.tight_layout(); plt.savefig(f"{out_prefix}_pr.png", dpi=150); plt.close()

def plot_calibration(y_true, probs, out_png, n_bins=10):
    y_pred = np.max(probs, axis=1)
    y_is_correct = (np.argmax(probs, axis=1) == y_true).astype(int)
    frac_pos, mean_pred = calibration_curve(y_is_correct, y_pred, n_bins=n_bins, strategy="quantile")
    ece = np.mean(np.abs(frac_pos - mean_pred))
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Predicted confidence"); plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability Diagram (ECE≈{ece:.3f})")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def dice_score(gt, pr, cls):
    gt1 = (gt == cls)
    pr1 = (pr == cls)
    inter = np.count_nonzero(gt1 & pr1)
    s = gt1.sum() + pr1.sum()
    return (2*inter / s) if s > 0 else (1.0 if gt1.sum() == pr1.sum() == 0 else 0.0)

def voxel_volume_mm3(img):
    hdr = img.header
    pixdim = hdr.get_zooms()
    v = 1.0
    for d in range(min(3, len(pixdim))):
        v *= float(pixdim[d])
    return v

def collect_seg_metrics(labels_dir, preds_dir, out_csv, classes=(1,2)):
    labels_dir, preds_dir = Path(labels_dir), Path(preds_dir)
    rows = []
    for gt_path in sorted(labels_dir.glob("*.nii*")):
        case = gt_path.name.replace(".nii.gz","").replace(".nii","")
        pr_path = preds_dir / f"{case}.nii.gz"
        if not pr_path.exists():
            pr_path = preds_dir / f"{case}.nii"
            if not pr_path.exists():
                continue
        gt_img = nib.load(str(gt_path)); gt = gt_img.get_fdata().astype(np.int32)
        pr_img = nib.load(str(pr_path)); pr = pr_img.get_fdata().astype(np.int32)
        vv = voxel_volume_mm3(gt_img)
        row = {"case": case}
        # volumes (ml)
        for c in classes:
            row[f"gt_vol_cls{c}_ml"] = gt[gt==c].size * vv / 1000.0
            row[f"pr_vol_cls{c}_ml"] = pr[pr==c].size * vv / 1000.0
            row[f"dice_cls{c}"] = dice_score(gt, pr, c)
        rows.append(row)
    # write CSV
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["case"])
        w.writeheader()
        for r in rows: w.writerow(r)
    return rows

def plot_seg_boxplots(rows, classes, out_png):
    data = []
    labels = []
    for c in classes:
        data.append([r[f"dice_cls{c}"] for r in rows])
        labels.append(f"Class {c}")
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Dice")
    plt.title("Segmentation Dice per class (validation)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_dice_vs_volume(rows, cls, out_png):
    dice = np.array([r[f"dice_cls{cls}"] for r in rows])
    gt_vol = np.array([r[f"gt_vol_cls{cls}_ml"] for r in rows])
    plt.figure()
    plt.scatter(gt_vol, dice, s=10)
    plt.xlabel(f"GT volume class {cls} (ml)"); plt.ylabel("Dice")
    plt.title(f"Dice vs. GT volume (class {cls})")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction folder (inference output)")
    ap.add_argument("--labels", required=True, help="ground-truth labels folder")
    ap.add_argument("--out", default=None, help="output folder for figures")
    ap.add_argument("--classes", default="1,2", help="segmentation classes to eval, comma-separated")
    args = ap.parse_args()

    # paths (ADD THESE)
    pkl_path = os.path.join(args.pred, "classification_results.pkl")
    csv_path = os.path.join(args.pred, "subtype_results.csv")

    out_dir = args.out or os.path.join(args.pred, "figs")
    ensure_dir(out_dir)

    # Try to infer class names from dataset.json if present
    class_names = None
    dj = Path(args.pred) / "dataset.json"
    if dj.exists():
        try:
            with open(dj) as f:
                djobj = json.load(f)
            labs = djobj.get("labels") or {}
            class_names = [labs.get(str(i), f"class{i}") for i in range(len(labs))]
        except Exception:
            pass
    if class_names is None:
        class_names = ["class0", "class1", "class2"]

    # ===== Classification plots =====
    # (CALL UNCONDITIONALLY; let loader fall back to CSV if needed)
    probs, labels, ids = load_cls_results(pkl_path, csv_path=csv_path)
    
    plot_confusion_simple(labels, probs, os.path.join(out_dir, "classification_confusion"), class_names)

    y_pred = np.argmax(probs, axis=1)

    plot_confusion(labels, y_pred, class_names[:probs.shape[1]],
                   os.path.join(out_dir, "classification_confusion.png"))
    plot_roc_pr(labels, probs, class_names[:probs.shape[1]],
                os.path.join(out_dir, "classification"))
    plot_calibration(labels, probs,
                     os.path.join(out_dir, "classification_calibration.png"))

    import pandas as pd
    topk = np.max(probs, axis=1)
    pd.DataFrame({"id": ids, "label": labels, "pred": y_pred, "conf": topk}) \
      .to_csv(os.path.join(out_dir, "classification_preds.csv"), index=False)

    # ===== Segmentation metrics & plots =====
    classes = tuple(int(x) for x in args.classes.split(","))
    rows = collect_seg_metrics(args.labels, args.pred,
                               os.path.join(out_dir, "segmentation_metrics.csv"),
                               classes=classes)
    if rows:
        plot_seg_boxplots(rows, classes,
                          os.path.join(out_dir, "segmentation_dice_boxplots.png"))
        for c in classes:
            plot_dice_vs_volume(rows, c,
                                os.path.join(out_dir, f"segmentation_dice_vs_volume_cls{c}.png"))

if __name__ == "__main__":
    main()
