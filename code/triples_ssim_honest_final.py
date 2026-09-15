"""Honest whole-pipeline SSIM impact for the triples category, across every
example. Same methodology as pairs_ssim_honest_final.py, generalized to 3
sequential corrupt/restore steps.

"Achieved" replays the model's full predicted pipeline (every type it named,
its own order, its own claimed method+value). "Ceiling" restores with the
true order/method/value for all three distortions.

Saves incrementally, one JSON line per example, so a dead/timed-out job can
just be resubmitted and pick up where it left off -- three sequential
restoration steps per example makes this the slowest of these scripts.
"""
import json
import re
import sys
import os
import statistics

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
OUT_JSONL = os.path.join(_REPO, "final_test/stats/results/triples_ssim_honest_deltas.jsonl")
OUT_SUMMARY = os.path.join(_REPO, "final_test/stats/results/triples_ssim_honest.json")

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
    n3 = [r for r in rows if len(r["true_types"]) == 3]
    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    base_to_image = {e["base_id"]: e["image"] for e in test_manifest}

    already_done = []
    if os.path.exists(OUT_JSONL):
        already_done = [json.loads(l) for l in open(OUT_JSONL) if l.strip()]
    n_done = len(already_done)
    if n_done > 0:
        print(f"Resuming: {n_done} already computed, skipping to example {n_done}.", flush=True)
    remaining = n3[n_done:]

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    out_f = open(OUT_JSONL, "a" if n_done > 0 else "w")

    for i, r in enumerate(remaining):
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        skipped_reason = None
        delta = None
        if img_rel is None:
            skipped_reason = "no_image_path"
        else:
            img = cv2.imread(os.path.join(RAW_DIR, img_rel))
            if img is None:
                skipped_reason = "image_read_failed"
            else:
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
                delta = ceiling_ssim - achieved_ssim

        out_f.write(json.dumps({"id": r["id"], "delta": delta, "skipped_reason": skipped_reason}) + "\n")
        out_f.flush()

        if (n_done + i + 1) % 100 == 0:
            print(f"  progress: {n_done + i + 1}/{len(n3)}", flush=True)

    out_f.close()

    all_rows = [json.loads(l) for l in open(OUT_JSONL) if l.strip()]
    deltas = [r["delta"] for r in all_rows if r["delta"] is not None]
    n_skipped = sum(1 for r in all_rows if r["delta"] is None)

    result = {
        "n_compared": len(deltas), "n_skipped": n_skipped,
        "mean_ssim_loss": statistics.mean(deltas) if deltas else None,
        "median_ssim_loss": statistics.median(deltas) if deltas else None,
        "fraction_negligible_loss_lt_0.01": sum(1 for d in deltas if d < 0.01) / len(deltas) if deltas else None,
    }
    print(f"compared={len(deltas)}, skipped={n_skipped}")
    if deltas:
        print(f"mean SSIM lost vs ceiling: {statistics.mean(deltas):.4f}")
        print(f"median SSIM lost vs ceiling: {statistics.median(deltas):.4f}")
        print(f"fraction negligible (<0.01): {result['fraction_negligible_loss_lt_0.01']:.3f}")

    with open(OUT_SUMMARY, "w") as f:
        json.dump(result, f, indent=2)

    # the incremental jsonl only stores id/delta to keep writes fast --
    # pull full context for just the top 10 worst from the answer rows
    by_id = {r["id"]: r for r in n3}
    worst_rows = sorted((r for r in all_rows if r["delta"] is not None), key=lambda r: -r["delta"])[:10]
    worst_cases = []
    for wr in worst_rows:
        r = by_id.get(wr["id"])
        if r is None:
            continue
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        worst_cases.append({
            "id": r["id"], "base_id": base,
            "image_path": os.path.join(RAW_DIR, img_rel) if img_rel else None,
            "image_path_relative": img_rel,
            "true_types": r["true_types"], "pred_types": r["pred_types"],
            "true_order": r["true_order"], "pred_order": r["pred_order"],
            "ssim_loss": wr["delta"], "output": r["output"],
        })
    if worst_cases:
        worst_cases_dir = os.path.join(_REPO, "final_test/stats/worst_cases")
        os.makedirs(worst_cases_dir, exist_ok=True)
        worst_path = os.path.join(worst_cases_dir, "triples_ssim_worst_cases.json")
        with open(worst_path, "w") as f:
            json.dump(worst_cases, f, indent=2)
        print(f"Saved -> {worst_path}")
    print(f"Saved -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
