"""Same honest whole-pipeline SSIM computation as pairs_ssim_honest_final.py,
broken down by pair-subset (gamma+jpeg / gamma+noise / jpeg+noise) instead of
pooled into one number.

Run via sbatch.
"""
import json
import re
import sys
import os
import statistics
import collections

import cv2
from skimage.metrics import structural_similarity as ssim

_REPO = "/nfsd/lttm4/tesisti/gramatchi"
sys.path.insert(0, os.path.join(_REPO, "v6"))
from methods_jpeg import METHODS as JPEG_METHODS
from methods_noise import METHODS as NOISE_METHODS
from methods_gamma import METHODS as GAMMA_METHODS

sys.path.insert(0, _REPO)
from corruptors import add_jpeg_compression, add_gaussian_noise, add_gamma_corruption

ANSWERS_PATH = os.path.join(_REPO, "final_test/eval_results/final_test_pipeline_answers.jsonl")
TEST_MANIFEST_PATH = os.path.join(_REPO, "final_test/dataset/llava_final_test.json")
RAW_DIR = "/home/gramatchin/data/raw"
OUT_PATH = os.path.join(_REPO, "final_test/stats/results/pairs_ssim_by_subset.json")

METHOD_FN = {
    "jpeg": {name: fn for name, fn in JPEG_METHODS},
    "noise": {name: fn for name, fn in NOISE_METHODS},
    "gamma": {name: fn for name, fn in GAMMA_METHODS},
}
CORRUPT_FN = {
    "jpeg": lambda img, v: add_jpeg_compression(img, quality=int(v))[0],
    "noise": lambda img, v: add_gaussian_noise(img, sigma=v)[0],
    "gamma": lambda img, v: add_gamma_corruption(img, gamma=v)[0],
}
ALL_METHOD_NAMES = sorted(
    set(m for m, _ in JPEG_METHODS) | set(m for m, _ in NOISE_METHODS) | set(m for m, _ in GAMMA_METHODS),
    key=len, reverse=True,
)
METHOD_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in ALL_METHOD_NAMES) + r")\b")
VALUE_RE = {
    "jpeg": re.compile(r"quality\D{0,10}?(\d+)", re.IGNORECASE),
    "noise": re.compile(r"sigma\D{0,10}?([0-9]*\.?[0-9]+)", re.IGNORECASE),
    "gamma": re.compile(r"gamma\D{0,5}?([0-9]*\.?[0-9]+)", re.IGNORECASE),
}


def claimed_value(output, t):
    m = VALUE_RE[t].search(output)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


def apply_pipeline(img, order, values_by_type, methods_by_type):
    cur = img
    for t in order:
        cur = METHOD_FN[t][methods_by_type[t]](cur, values_by_type[t])
    return cur


def corrupt_sequence(img, order, values_by_type):
    cur = img
    for t in order:
        cur = CORRUPT_FN[t](cur, values_by_type[t])
    return cur


def main():
    rows = [json.loads(l) for l in open(ANSWERS_PATH)]
    n2 = [r for r in rows if len(r["true_types"]) == 2]
    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    base_to_image = {e["base_id"]: e["image"] for e in test_manifest}

    by_subset = collections.defaultdict(list)
    n_skipped = collections.Counter()
    n_processed = 0
    for r in n2:
        subset = tuple(sorted(r["true_types"]))
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        if img_rel is None:
            n_skipped[subset] += 1
            continue
        img = cv2.imread(os.path.join(RAW_DIR, img_rel))
        if img is None:
            n_skipped[subset] += 1
            continue

        true_values = {t: r["steps"][t]["true_value"] for t in r["true_types"]}
        true_methods = {t: r["steps"][t]["true_method"] for t in r["true_types"]}
        corruption_order = list(reversed(r["true_order"]))
        corrupted = corrupt_sequence(img, corruption_order, true_values)
        gt_restored = apply_pipeline(corrupted, r["true_order"], true_values, true_methods)
        ceiling_ssim = _ssim(img, gt_restored)

        pred_order = r["pred_order"]
        method_mentions = [m.group(1) for m in METHOD_RE.finditer(r["output"])]
        pred_method_by_type = dict(zip(pred_order, method_mentions))
        pred_value_by_type = {t: claimed_value(r["output"], t) for t in ["jpeg", "noise", "gamma"]}

        cur = corrupted
        for t in pred_order:
            pm = pred_method_by_type.get(t)
            pv = pred_value_by_type.get(t)
            if pm not in METHOD_FN[t] or pv is None or (t == "gamma" and pv <= 0):
                continue
            try:
                cur = METHOD_FN[t][pm](cur, pv)
            except Exception:
                continue
        achieved_ssim = _ssim(img, cur)

        by_subset[subset].append(ceiling_ssim - achieved_ssim)
        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  progress: {n_processed}/{len(n2)}", flush=True)

    result = {}
    for subset in sorted(by_subset):
        deltas = by_subset[subset]
        key = "+".join(subset)
        result[key] = {
            "n": len(deltas), "n_skipped": n_skipped[subset],
            "mean_ssim_loss": statistics.mean(deltas),
            "median_ssim_loss": statistics.median(deltas),
            "fraction_negligible_loss_lt_0.01": sum(1 for d in deltas if d < 0.01) / len(deltas),
        }
        print(f"{key}: n={len(deltas)}, skipped={n_skipped[subset]}")
        print(f"  mean SSIM lost: {statistics.mean(deltas):.4f}")
        print(f"  median SSIM lost: {statistics.median(deltas):.4f}")
        print(f"  fraction negligible (<0.01): {result[key]['fraction_negligible_loss_lt_0.01']:.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
