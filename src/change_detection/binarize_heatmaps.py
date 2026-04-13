"""
Heatmap evaluation and mask generation script.

Modes (combinable):
  --verify      : Load saved NPZ heatmaps, compute metrics, compare against existing CSV.
  --gen-masks   : Calibrate a threshold per (object, general_defect) group using GT samples,
                  then binarize ALL heatmaps (with or without GT) and save as PNG masks.
  --eval-masks  : Evaluate saved binary masks in --masks-dir-name against GT masks.
                  Computes per-category F1, IoU, and pixel accuracy, prints summary table.

Threshold calibration:
  For each (object, prompt) group, compute the best-F1 threshold on each GT sample,
  average them → group threshold. Falls back to Otsu if no GT is available for a group.

Output masks are saved to:
  {dataset_root}/{object}/test/general/{masks_dir_name}/{img_stem}_mask.png

Usage:
    python src/change_detection/binarize_heatmaps.py \
        --output-dir outputs/mvtec/gdino_diff__yoloe_gating_sharp_fuse_a0.80_layers_2_3_4_5_6 \
        --dataset-path mvtec \
        --gen-masks --eval-masks
"""

import os
import csv
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


#  Metric helpers


def best_f1(heatmap, gt_binary, n_thresholds=50):
    gt_flat = gt_binary.flatten().astype(bool)
    best_f1_val, best_thresh = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, n_thresholds):
        pred = (heatmap > t).flatten()
        tp = (pred & gt_flat).sum()
        fp = (pred & ~gt_flat).sum()
        fn = (~pred & gt_flat).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = float(2 * prec * rec / (prec + rec + 1e-8))
        if f1 > best_f1_val:
            best_f1_val = f1
            best_thresh = t
    return best_f1_val, best_thresh


def otsu_threshold(heatmap):
    hm_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_val / 255.0


#  Sample collection


def extract_general_defect(defect_syn):
    clean = defect_syn.strip().strip('"')
    if "based on this image." in clean:
        clean = clean.split("based on this image.")[-1].strip()
    return clean.split(":")[0].strip()


def collect_samples(csv_path, dataset_root, ref_root, npz_dir):
    """
    Read the tracking CSV and collect all samples, attaching NPZ and GT mask paths.
    Returns a list of dicts.
    """
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Collecting samples"):
            obj = row["object"]
            img_path = row["img_path"]
            base_img_path = row["base_img_path"]
            defect_syn = row["defect_synonym"].strip().strip('"')

            anom_full = os.path.join(dataset_root, img_path)
            base_full = os.path.join(ref_root, base_img_path)
            img_stem = Path(img_path).stem

            npz_path = os.path.join(npz_dir, obj, f"{img_stem}.npz")
            mask_path = os.path.join(
                dataset_root,
                os.path.dirname(img_path),
                "masks_modify_total",
                f"{img_stem}_mask.png",
            )

            has_npz = os.path.exists(npz_path)
            has_gt = os.path.exists(mask_path)
            img_exists = os.path.exists(anom_full)

            general_defect = extract_general_defect(defect_syn)

            samples.append(
                {
                    "object": obj,
                    "img_path": img_path,
                    "img_stem": img_stem,
                    "anom_img": anom_full,
                    "base_img": base_full,
                    "mask": mask_path if has_gt else None,
                    "npz_path": npz_path,
                    "general_defect": general_defect,
                    "has_npz": has_npz,
                    "has_gt": has_gt,
                    "img_exists": img_exists,
                }
            )
    return samples


#  Verify mode


def verify_metrics(samples, existing_csv_path):
    """
    For each sample that has both an NPZ and a GT mask, compute metrics and compare
    against the existing CSV produced by grounding_dino_diff.py.
    Prints a summary of discrepancies.
    """
    # Load existing CSV keyed by (object, img_name)
    existing = {}
    fieldnames = []
    if os.path.exists(existing_csv_path):
        with open(existing_csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                existing[(row["object"], row["img_name"])] = row
    else:
        print(f"[verify] CSV not found: {existing_csv_path}")

    # Detect which AUC column corresponds to the saved (final) heatmap.
    # So the saved NPZ contains fused_sharp > fused_geom > backbone, in that order.
    auc_col = None
    _priority = ["auc_fused_sharp", "auc_fused_geom"]
    for candidate in _priority:
        if candidate in fieldnames:
            auc_col = candidate
            break
    if auc_col is None:
        # Fall back: first auc_ col that looks like a raw backbone score
        for fn in fieldnames:
            if fn.startswith("auc_") and "absdiff" not in fn and "fused" not in fn:
                auc_col = fn
                break

    print(f"\n[verify] Comparing against: {existing_csv_path}")
    print(f"[verify] AUC column used for comparison: {auc_col}")

    mismatches = 0
    checked = 0
    auc_diffs = []

    # Only process samples with both NPZ and GT
    verify_samples = [s for s in samples if s["has_npz"] and s["has_gt"]]
    print(f"[verify] Samples to check: {len(verify_samples)}")

    for s in tqdm(verify_samples, desc="Verifying"):
        heatmap = np.load(s["npz_path"])["heatmap"]

        gt_mask_arr = np.array(Image.open(s["mask"]).convert("L"))
        gt_binary = (gt_mask_arr > 127).astype(np.uint8)

        # Skip trivial GT
        if gt_binary.sum() == 0 or gt_binary.sum() == gt_binary.size:
            continue

        # Resize heatmap to match GT if needed
        if heatmap.shape != gt_mask_arr.shape:
            heatmap = cv2.resize(
                heatmap,
                (gt_mask_arr.shape[1], gt_mask_arr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        gt_flat = gt_binary.flatten()
        hm_flat = heatmap.flatten()

        try:
            auc = roc_auc_score(gt_flat, hm_flat)
            ap = average_precision_score(gt_flat, hm_flat)
        except ValueError:
            continue

        f1_val, _ = best_f1(heatmap, gt_binary)

        checked += 1
        key = (s["object"], s["img_stem"])

        if key in existing and auc_col:
            try:
                csv_auc = float(existing[key][auc_col])
                diff = abs(auc - csv_auc)
                auc_diffs.append(diff)
                if diff > 0.01:
                    mismatches += 1
                    print(
                        f"  MISMATCH {s['object']}/{s['img_stem']}: "
                        f"recomputed={auc:.4f}  csv={csv_auc:.4f}  diff={diff:.4f}"
                    )
            except (KeyError, ValueError):
                pass
        else:
            # Sample not in CSV yet (still running) - just report metrics
            if checked <= 5:
                print(
                    f"  {s['object']}/{s['img_stem']}: AUC={auc:.4f} AP={ap:.4f} F1={f1_val:.4f}"
                )

    if auc_diffs:
        print(
            f"\n[verify] Checked {checked} samples. "
            f"Mean AUC diff={np.mean(auc_diffs):.5f}, "
            f"Max={np.max(auc_diffs):.5f}, "
            f"Mismatches (>0.01): {mismatches}"
        )
    else:
        print(f"\n[verify] Checked {checked} samples. No CSV entries to compare yet.")


#  Threshold calibration and mask generation


THRESH_KEY_SEP = "|"


def thresholds_to_dict(thresholds):
    """Convert (obj, gd) keyed dict to JSON-serialisable string-keyed dict."""
    return {f"{obj}{THRESH_KEY_SEP}{gd}": t for (obj, gd), t in thresholds.items()}


def dict_to_thresholds(d):
    """Reverse of thresholds_to_dict."""
    result = {}
    for key, t in d.items():
        obj, gd = key.split(THRESH_KEY_SEP, 1)
        result[(obj, gd)] = float(t)
    return result


def save_thresholds(thresholds, path):
    with open(path, "w") as f:
        json.dump(thresholds_to_dict(thresholds), f, indent=2)
    print(f"[calibrate] Thresholds saved to {path}")


def load_thresholds(path):
    with open(path) as f:
        d = json.load(f)
    thresholds = dict_to_thresholds(d)
    print(f"[calibrate] Loaded {len(thresholds)} thresholds from {path}")
    return thresholds


def calibrate_thresholds(samples, save_path=None):
    """
    For each (object, general_defect) group, compute the mean best-F1 threshold
    over all GT samples that have an NPZ. Falls back to None if no GT available.

    Returns: dict[(object, general_defect)] -> float threshold
    Optionally saves to save_path as JSON.
    """
    # Group samples that have both NPZ and GT
    groups = defaultdict(list)
    for s in samples:
        if s["has_npz"] and s["has_gt"]:
            groups[(s["object"], s["general_defect"])].append(s)

    thresholds = {}
    print(f"\n[calibrate] {len(groups)} groups have GT+NPZ samples.")

    for (obj, gd), group_samples in tqdm(groups.items(), desc="Calibrating"):
        best_threshs = []
        for s in group_samples:
            heatmap = np.load(s["npz_path"])["heatmap"]
            gt_mask_arr = np.array(Image.open(s["mask"]).convert("L"))
            gt_binary = (gt_mask_arr > 127).astype(np.uint8)

            if gt_binary.sum() == 0 or gt_binary.sum() == gt_binary.size:
                continue

            if heatmap.shape != gt_mask_arr.shape:
                heatmap = cv2.resize(
                    heatmap,
                    (gt_mask_arr.shape[1], gt_mask_arr.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            _, t = best_f1(heatmap, gt_binary)
            best_threshs.append(t)

        if best_threshs:
            calibrated = float(np.mean(best_threshs))
            thresholds[(obj, gd)] = calibrated

    if save_path:
        save_thresholds(thresholds, save_path)

    return thresholds


def generate_masks(
    samples, thresholds, dataset_root, masks_dir_name, dry_run=False, resume=False
):
    """
    For each sample with an NPZ, binarize the heatmap using the calibrated threshold
    (or Otsu fallback) and save as a PNG mask.

    Mask path: {dataset_root}/{img_dir}/{masks_dir_name}/{img_stem}_mask.png

    resume: skip samples whose output mask already exists.
    """
    created = 0
    skipped_no_npz = sum(1 for s in samples if not s["has_npz"])
    skipped_resume = 0
    otsu_fallback = 0

    # Collect all (object, general_defect) combos with no GT at all for reporting
    all_groups = set(
        (s["object"], s["general_defect"]) for s in samples if s["has_npz"]
    )
    no_gt_groups = all_groups - set(thresholds.keys())
    if no_gt_groups:
        print(
            f"\n[gen-masks] {len(no_gt_groups)} groups have no GT → will use Otsu fallback:"
        )
        for g in sorted(no_gt_groups):
            print(f"  {g[0]} / {g[1]}")

    npz_samples = [s for s in samples if s["has_npz"]]
    for s in tqdm(npz_samples, desc="Generating masks"):
        img_dir = os.path.dirname(os.path.join(dataset_root, s["img_path"]))
        masks_out_dir = os.path.join(img_dir, masks_dir_name)
        mask_out_path = os.path.join(masks_out_dir, f"{s['img_stem']}_mask.png")

        if resume and os.path.exists(mask_out_path):
            skipped_resume += 1
            continue

        heatmap = np.load(s["npz_path"])["heatmap"]

        # Get threshold
        group_key = (s["object"], s["general_defect"])
        if group_key in thresholds:
            thresh = thresholds[group_key]
        else:
            thresh = otsu_threshold(heatmap)
            otsu_fallback += 1

        binary_mask = (heatmap > thresh).astype(np.uint8) * 255

        if not dry_run:
            os.makedirs(masks_out_dir, exist_ok=True)
            Image.fromarray(binary_mask).save(mask_out_path)

        created += 1

    print(
        f"\n[gen-masks] Done. "
        f"Masks created: {created}, "
        f"Otsu fallback: {otsu_fallback}, "
        f"Skipped (resume): {skipped_resume}, "
        f"Skipped (no NPZ): {skipped_no_npz}"
    )
    if dry_run:
        print("[gen-masks] DRY RUN : no files were written.")


#  Mask evaluation


def eval_masks(samples, dataset_root, masks_dir_name):
    """
    Compare binary masks in masks_dir_name against GT masks.
    Computes F1, IoU, pixel accuracy per sample, then summarises per category.
    Only processes samples that have a GT mask; reports how many predicted masks exist.
    """
    gt_samples = [s for s in samples if s["has_gt"]]
    print(f"\n[eval-masks] Evaluating against GT : {len(gt_samples)} GT samples")

    rows = []
    missing_pred = 0

    for s in tqdm(gt_samples, desc="Evaluating masks"):
        img_dir = os.path.dirname(os.path.join(dataset_root, s["img_path"]))
        pred_path = os.path.join(img_dir, masks_dir_name, f"{s['img_stem']}_mask.png")

        if not os.path.exists(pred_path):
            missing_pred += 1
            continue

        gt_arr = np.array(Image.open(s["mask"]).convert("L"))
        pred_arr = np.array(Image.open(pred_path).convert("L"))

        gt_bin = (gt_arr > 127).astype(bool)
        if gt_bin.sum() == 0 or gt_bin.sum() == gt_bin.size:
            continue  # skip trivial GT

        # Resize pred to GT size if needed
        if pred_arr.shape != gt_arr.shape:
            pred_arr = np.array(
                Image.fromarray(pred_arr).resize(
                    (gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST
                )
            )

        pred_bin = (pred_arr > 127).astype(bool)

        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & ~gt_bin).sum())
        fn = int((~pred_bin & gt_bin).sum())
        tn = int((~pred_bin & ~gt_bin).sum())

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = float(2 * prec * rec / (prec + rec + 1e-8))
        iou = float(tp / (tp + fp + fn + 1e-8))
        acc = float((tp + tn) / (tp + tn + fp + fn + 1e-8))

        rows.append(
            {
                "object": s["object"],
                "img_stem": s["img_stem"],
                "general_defect": s["general_defect"],
                "f1": f1,
                "iou": iou,
                "acc": acc,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    if missing_pred:
        print(
            f"  [warn] {missing_pred} GT samples have no predicted mask in '{masks_dir_name}/'"
        )

    if not rows:
        print("  [eval-masks] No valid samples to evaluate.")
        return

    # Per-category summary
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["object"]].append(r)

    metrics = [("f1", "F1"), ("iou", "IoU"), ("acc", "PixAcc")]

    print(f"\n{'=' * 65}")
    print(f"Mask evaluation: {masks_dir_name}  ({len(rows)} samples)")
    print(f"{'=' * 65}")
    print(f"{'Category':<15} {'N':>4}  {'F1':>7}  {'IoU':>7}  {'PixAcc':>8}")
    print("-" * 65)

    all_vals = defaultdict(list)
    for obj in sorted(by_cat.keys()):
        cr = by_cat[obj]
        f1_m = float(np.mean([r["f1"] for r in cr]))
        iou_m = float(np.mean([r["iou"] for r in cr]))
        acc_m = float(np.mean([r["acc"] for r in cr]))
        print(f"{obj:<15} {len(cr):>4}  {f1_m:>7.4f}  {iou_m:>7.4f}  {acc_m:>8.4f}")
        for key in ("f1", "iou", "acc"):
            all_vals[key].extend([r[key] for r in cr])

    print("-" * 65)
    print(
        f"{'OVERALL':<15} {len(rows):>4}  "
        f"{np.mean(all_vals['f1']):>7.4f}  "
        f"{np.mean(all_vals['iou']):>7.4f}  "
        f"{np.mean(all_vals['acc']):>8.4f}"
    )
    print(f"{'=' * 65}")

    # Per-prompt breakdown (optional verbose)
    print(f"\n{'' * 65}")
    print("Per-prompt breakdown")
    print(f"{'' * 65}")
    print(f"{'Object':<15} {'Prompt':<30} {'N':>4}  {'F1':>7}  {'IoU':>7}")
    print("-" * 65)
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[(r["object"], r["general_defect"])].append(r)
    for obj, gd in sorted(by_prompt.keys()):
        cr = by_prompt[(obj, gd)]
        f1_m = float(np.mean([r["f1"] for r in cr]))
        iou_m = float(np.mean([r["iou"] for r in cr]))
        print(f"{obj:<15} {gd:<30} {len(cr):>4}  {f1_m:>7.4f}  {iou_m:>7.4f}")


#  Summary


def print_threshold_summary(thresholds, samples):
    """Print a table of calibrated thresholds per (object, prompt)."""
    print(f"\n{'=' * 70}")
    print("Calibrated thresholds per (object, general_defect)")
    print(f"{'=' * 70}")
    print(f"{'Object':<15} {'Prompt':<35} {'Thresh':>7} {'N_GT':>5}")
    print("-" * 70)

    # Count GT samples per group
    gt_counts = defaultdict(int)
    for s in samples:
        if s["has_npz"] and s["has_gt"]:
            gt_counts[(s["object"], s["general_defect"])] += 1

    for obj, gd in sorted(thresholds.keys()):
        t = thresholds[(obj, gd)]
        n = gt_counts[(obj, gd)]
        print(f"{obj:<15} {gd:<35} {t:>7.4f} {n:>5}")

    print(f"\nTotal groups calibrated: {len(thresholds)}")


#  Main


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved heatmaps and/or generate binary masks."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the experiment output directory (contains heatmaps/ and metrics.csv). "
        "Can be relative to PROJECT_ROOT.",
    )
    parser.add_argument(
        "--dataset-path",
        default="visa",
        help="Dataset name: visa or mvtec (default: visa)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compute metrics on saved heatmaps and compare with existing CSV.",
    )
    parser.add_argument(
        "--gen-masks",
        action="store_true",
        help="Calibrate thresholds and generate binary masks for all heatmaps.",
    )
    parser.add_argument(
        "--masks-dir-name",
        default="our_masks",
        help="Name of the output masks subdirectory (default: our_masks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --gen-masks: compute thresholds and report but don't write mask files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --gen-masks: skip samples whose output mask already exists.",
    )
    parser.add_argument(
        "--eval-masks",
        action="store_true",
        help="Evaluate binary masks in --masks-dir-name against GT: F1, IoU, pixel accuracy.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Path to a thresholds JSON file (skips calibration). "
        "If not set and --gen-masks is used, calibration runs automatically. "
        "Defaults to <output-dir>/thresholds.json if it exists.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Force threshold calibration even if a thresholds JSON already exists.",
    )
    args = parser.parse_args()

    if not args.verify and not args.gen_masks and not args.eval_masks:
        parser.error("Specify at least one of --verify, --gen-masks, --eval-masks")

    # Resolve output dir
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    output_dir = os.path.abspath(output_dir)

    npz_dir = os.path.join(output_dir, "heatmaps")
    metrics_csv = os.path.join(output_dir, "metrics.csv")

    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"Heatmaps dir not found: {npz_dir}")

    # Dataset paths
    dataset_name = args.dataset_path
    dataset_root = os.path.join(
        PROJECT_ROOT, "datasets", f"all_generated_{dataset_name}"
    )
    ref_root = os.path.join(PROJECT_ROOT, "datasets", dataset_name)
    csv_tracking = os.path.join(dataset_root, "generation_tracking.csv")

    print(f"Output dir  : {output_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Heatmaps dir: {npz_dir}")

    # Collect samples
    samples = collect_samples(csv_tracking, dataset_root, ref_root, npz_dir)
    total = len(samples)
    n_npz = sum(1 for s in samples if s["has_npz"])
    n_gt = sum(1 for s in samples if s["has_gt"])
    n_both = sum(1 for s in samples if s["has_npz"] and s["has_gt"])
    print(
        f"\nSamples: {total} total | {n_npz} with NPZ | {n_gt} with GT | {n_both} with both"
    )

    if args.verify:
        verify_metrics(samples, metrics_csv)

    if args.gen_masks:
        default_thresh_path = os.path.join(output_dir, "thresholds.json")

        # Resolve thresholds: load from file or calibrate
        if args.calibrate:
            thresholds = calibrate_thresholds(samples, save_path=default_thresh_path)
        elif args.thresholds and os.path.exists(args.thresholds):
            thresholds = load_thresholds(args.thresholds)
        elif os.path.exists(default_thresh_path) and not args.calibrate:
            thresholds = load_thresholds(default_thresh_path)
        else:
            thresholds = calibrate_thresholds(samples, save_path=default_thresh_path)

        print_threshold_summary(thresholds, samples)
        generate_masks(
            samples,
            thresholds,
            dataset_root,
            args.masks_dir_name,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        print(
            f"\nMasks saved to: {dataset_root}/<object>/test/general/{args.masks_dir_name}/"
        )

    if args.eval_masks:
        eval_masks(samples, dataset_root, args.masks_dir_name)


if __name__ == "__main__":
    main()
