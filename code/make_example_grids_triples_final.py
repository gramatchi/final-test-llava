"""Renders 3-panel example images for the triples category: 50 from the
flagged pool + up to 20 fully-correct examples + the worst by SSIM loss.

"Restored" replays the model's full predicted pipeline: every type it named,
its own order, its own claimed method+value.

Saved under final_test/example_images/triples/{sample,correct,worst_ssim}/.
"""
import json
import os
import random
import re
import sys
import textwrap

import cv2
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

_REPO = "/nfsd/lttm4/tesisti/gramatchi"
sys.path.insert(0, os.path.join(_REPO, "v6"))
from methods_jpeg import METHODS as JPEG_METHODS
from methods_noise import METHODS as NOISE_METHODS
from methods_gamma import METHODS as GAMMA_METHODS

sys.path.insert(0, _REPO)
from corruptors import add_jpeg_compression, add_gaussian_noise, add_gamma_corruption

ANSWERS_PATH = os.path.join(_REPO, "final_test/eval_results/final_test_pipeline_answers.jsonl")
FLAGGED_PATH = os.path.join(_REPO, "final_test/stats/results/triples_flagged_examples.json")
WORST_CASES_PATH = os.path.join(_REPO, "final_test/stats/worst_cases/triples_ssim_worst_cases.json")
TEST_MANIFEST_PATH = os.path.join(_REPO, "final_test/dataset/llava_final_test.json")
RAW_DIR = "/home/gramatchin/data/raw"
OUT_DIR = os.path.join(_REPO, "final_test/example_images/triples")
os.makedirs(os.path.join(OUT_DIR, "sample"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "worst_ssim"), exist_ok=True)

FONT_PATH = "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"
FONT = ImageFont.truetype(FONT_PATH, 15)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", 15)

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

PANEL_SIZE = 340
CAPTION_H = 300
MARGIN = 16
LINE_H = 18
WRAP_WIDTH = 46


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


def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def draw_caption(draw, x0, y0, w, h, lines, font=FONT):
    y = y0 + 6
    for line in lines:
        wrapped = textwrap.wrap(line, width=WRAP_WIDTH) or [""]
        for wline in wrapped:
            if y > y0 + h - LINE_H:
                draw.text((x0 + 6, y), "...", fill=(150, 0, 0), font=font)
                return
            draw.text((x0 + 6, y), wline, fill=(20, 20, 20), font=font)
            y += LINE_H


def make_panel_image(orig_bgr, corrupted_bgr, restored_bgr, left_lines, center_lines, right_lines, out_path):
    W = PANEL_SIZE * 3 + MARGIN * 4
    H = PANEL_SIZE + CAPTION_H + MARGIN * 2
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    panels = [orig_bgr, corrupted_bgr, restored_bgr]
    titles = ["ORIGINAL", "CORRUPTED", "RESTORED"]
    line_sets = [left_lines, center_lines, right_lines]
    for i, (panel_bgr, title, lines) in enumerate(zip(panels, titles, line_sets)):
        x0 = MARGIN + i * (PANEL_SIZE + MARGIN)
        y0 = MARGIN
        pil_img = bgr_to_pil(panel_bgr).resize((PANEL_SIZE, PANEL_SIZE))
        canvas.paste(pil_img, (x0, y0))
        draw.rectangle([x0, y0, x0 + PANEL_SIZE, y0 + PANEL_SIZE], outline=(0, 0, 0), width=1)
        draw.text((x0 + 6, y0 + 4), title, fill=(255, 0, 0), font=FONT_BOLD)
        draw_caption(draw, x0, y0 + PANEL_SIZE, PANEL_SIZE, CAPTION_H, lines)
    canvas.save(out_path)


def sev_word_hint(t, v):
    if t == "jpeg":
        return f"quality={int(v)}"
    if t == "noise":
        return f"sigma≈{v}"
    if t == "gamma":
        return f"gamma≈{v}"
    return str(v)


def render(r, entry, subdir, out_dir):
    img_rel = entry["image"]
    img = cv2.imread(os.path.join(RAW_DIR, img_rel))
    if img is None:
        print("  skip (image not found):", img_rel)
        return

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
    applied_steps = []
    for t in pred_order:
        pm = pred_method_by_type.get(t)
        pv = pred_value_by_type.get(t)
        if pm not in METHOD_FN[t] or pv is None or (t == "gamma" and pv <= 0):
            applied_steps.append(f"{t}: SKIPPED (no usable method/value)")
            continue
        try:
            cur = METHOD_FN[t][pm](cur, pv)
            applied_steps.append(f"{t}: '{pm}' @ {pv}")
        except Exception:
            applied_steps.append(f"{t}: SKIPPED (restore failed)")
    restored = cur
    achieved_ssim = _ssim(img, restored)
    ssim_corrupted = _ssim(img, corrupted)

    question = entry["conversations"][0]["value"].replace("<image>", "").strip()
    applied_desc = " -> ".join(f"{t} {sev_word_hint(t, true_values[t])}" for t in corruption_order)
    left_lines = [f'Question (as asked): "{question}"', f"(applied, in order: {applied_desc})"]
    center_lines = [f'Model answer (full): "{r["output"]}"']
    right_lines = [
        "SSIM original: 1.000",
        f"SSIM corrupted: {ssim_corrupted:.3f}",
        f"SSIM restored (model's full pipeline): {achieved_ssim:.3f}",
        f"  steps applied: {'; '.join(applied_steps)}",
        f"SSIM ideal (true pipeline): {ceiling_ssim:.3f}",
    ]

    out_path = os.path.join(out_dir, subdir, f"{r['id']}.png")
    make_panel_image(img, corrupted, restored, left_lines, center_lines, right_lines, out_path)


def main():
    # the fully-correct pool tends to be small here (needs all 3 types, full
    # order, and all 3 methods+values right at once) -- just take what's there
    rows = [json.loads(l) for l in open(ANSWERS_PATH)]
    n3 = [r for r in rows if len(r["true_types"]) == 3]
    by_id_full = {r["id"]: r for r in n3}

    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    by_id_entry = {e["id"]: e for e in test_manifest}

    flagged = json.load(open(FLAGGED_PATH))
    flagged_ids = [f["id"] for f in flagged]
    print(f"flagged (wrong) triples examples available: {len(flagged_ids)}")

    fully_correct = [
        r for r in n3
        if r["type_set_correct"] and r["order_correct"]
        and all(r["steps"][t]["method_correct"] and r["steps"][t]["value_correct"] for t in r["true_types"])
    ]
    print(f"fully-correct pool: {len(fully_correct)}")

    rng = random.Random(42)
    sample_ids = rng.sample(flagged_ids, min(50, len(flagged_ids)))
    correct_pick = rng.sample(fully_correct, min(20, len(fully_correct)))
    print(f"picked: {len(sample_ids)} sample (from flagged) + {len(correct_pick)} correct")

    for fid in sample_ids:
        render(by_id_full[fid], by_id_entry[fid], "sample", OUT_DIR)
    for r in correct_pick:
        render(r, by_id_entry[r["id"]], "correct", OUT_DIR)

    print(f"Saved -> {OUT_DIR}/sample/ ({len(sample_ids)} images)")
    print(f"Saved -> {OUT_DIR}/correct/ ({len(correct_pick)} images)")

    # ---- worst-SSIM cases (from triples_ssim_honest_final.py) ----
    if os.path.exists(WORST_CASES_PATH):
        worst_cases = json.load(open(WORST_CASES_PATH))
        print(f"triples worst-SSIM cases: {len(worst_cases)}")
        for c in worst_cases:
            r = by_id_full.get(c["id"])
            entry = by_id_entry.get(c["id"])
            if r is None or entry is None:
                print("  skip (not found in answers/manifest):", c["id"])
                continue
            render(r, entry, "worst_ssim", OUT_DIR)
        print(f"Saved -> {OUT_DIR}/worst_ssim/ ({len(worst_cases)} images)")
    else:
        print(f"(skipped worst_ssim gallery -- {WORST_CASES_PATH} not found yet)")


if __name__ == "__main__":
    main()
