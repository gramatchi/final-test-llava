"""Renders 3-panel (original | corrupted | restored) example images for the
singles category: 50 from the flagged (wrong-answer) pool + 20 fully-correct
examples + the 10 worst by SSIM loss.

Each PNG has 3 panels with a caption strip under each: the question + applied
distortion, the model's full answer, and SSIM numbers for corrupted/restored/
ideal. If the model gave no usable prediction, "restored" is just the
corrupted image, same convention as the SSIM-impact stats.

Saved under final_test/example_images/singles/{sample,correct,worst_ssim}/.
"""
import json
import os
import random
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
FLAGGED_PATH = os.path.join(_REPO, "final_test/stats/results/singles_flagged_examples.json")
WORST_CASES_PATH = os.path.join(_REPO, "final_test/stats/worst_cases/singles_ssim_worst_cases.json")
TEST_MANIFEST_PATH = os.path.join(_REPO, "final_test/dataset/llava_final_test.json")
RAW_DIR = "/home/gramatchin/data/raw"
OUT_DIR = os.path.join(_REPO, "final_test/example_images/singles")
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

PANEL_SIZE = 340
CAPTION_H = 260
MARGIN = 16
LINE_H = 18
WRAP_WIDTH = 46


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


BATCH_SUBDIR = "extra_images/0004000"


def _relative_path_for(bare_name):
    return f"{BATCH_SUBDIR}/{bare_name}"


def load_ssim_ceiling_lookup(dtype, value_field):
    # best achievable SSIM per photo/severity -- shown as the 4th ("ideal") number in each caption
    raw = json.load(open(os.path.join(_REPO, f"final_test/dataset/{dtype}_per_image/per_image.json")))
    out = {}
    for entry in raw:
        path = _relative_path_for(entry["image"])
        out[path] = {v: d["best_ssim"] for v, d in entry[f"by_{value_field}"].items()}
    return out


CEILING_LOOKUP = {
    "jpeg": load_ssim_ceiling_lookup("jpeg", "quality"),
    "noise": load_ssim_ceiling_lookup("noise", "sigma"),
}


def ceiling_ssim_for(t, img, corrupted, img_rel, true_v):
    # gamma's ceiling is computed on the fly (lut is the exact inverse);
    # jpeg/noise come from the offline per-image search
    if t == "gamma":
        return _ssim(img, METHOD_FN["gamma"]["lut"](corrupted, true_v))
    key = str(int(true_v)) if t == "jpeg" else str(true_v)
    entry = CEILING_LOOKUP[t].get(img_rel, {}).get(key)
    return entry["ssim"] if entry else None


def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def draw_caption(draw, x0, y0, w, h, lines, font=FONT):
    # wraps and draws each line, cutting off with "..." if it overflows the box
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
    t = r["true_types"][0]
    img_rel = entry["image"]
    img = cv2.imread(os.path.join(RAW_DIR, img_rel))
    if img is None:
        print("  skip (image not found):", img_rel)
        return
    true_v = r["steps"][t]["true_value"]
    pred_method = r["steps"][t]["pred_method"]
    pred_v = r["steps"][t]["pred_value"]

    corrupted = CORRUPT_FN[t](img, true_v)
    has_prediction = pred_method in METHOD_FN[t] and pred_v is not None and not (t == "gamma" and pred_v <= 0)
    restore_note = ""
    if has_prediction:
        try:
            restored = METHOD_FN[t][pred_method](corrupted, pred_v)
        except Exception as e:
            has_prediction = False
    if not has_prediction:
        restored = corrupted
        restore_note = " (no restoration applied -- model gave no usable method/value for this type)"

    ssim_corrupted = _ssim(img, corrupted)
    ssim_restored = _ssim(img, restored)
    ssim_ideal = ceiling_ssim_for(t, img, corrupted, img_rel, true_v)

    question = entry["conversations"][0]["value"].replace("<image>", "").strip()
    left_lines = [f'Question (as asked): "{question}"', f"(applied: {t}, {sev_word_hint(t, true_v)})"]
    center_lines = [f'Model answer (full): "{r["output"]}"']
    right_lines = [
        "SSIM original: 1.000",
        f"SSIM corrupted: {ssim_corrupted:.3f}",
        f"SSIM restored (model's method): {ssim_restored:.3f}{restore_note}",
        f"SSIM ideal (best possible method): {ssim_ideal:.3f}" if ssim_ideal is not None else "SSIM ideal: n/a",
    ]

    out_path = os.path.join(out_dir, subdir, f"{r['id']}.png")
    make_panel_image(img, corrupted, restored, left_lines, center_lines, right_lines, out_path)


def main():
    rows = [json.loads(l) for l in open(ANSWERS_PATH)]
    n1 = [r for r in rows if len(r["true_types"]) == 1]
    by_id_full = {r["id"]: r for r in n1}

    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    by_id_entry = {e["id"]: e for e in test_manifest}

    flagged = json.load(open(FLAGGED_PATH))
    flagged_ids = [f["id"] for f in flagged]
    print(f"flagged (wrong) singles examples available: {len(flagged_ids)}")

    fully_correct = []
    for r in n1:
        t = r["true_types"][0]
        if r["type_set_correct"] and r["steps"][t]["method_correct"] and r["steps"][t]["value_correct"]:
            fully_correct.append(r)
    print(f"fully-correct pool: {len(fully_correct)}")

    rng = random.Random(42)
    sample_ids = rng.sample(flagged_ids, min(50, len(flagged_ids)))
    correct_pick = rng.sample(fully_correct, min(20, len(fully_correct)))

    print(f"picked: {len(sample_ids)} sample (from flagged) + {len(correct_pick)} correct")

    for fid in sample_ids:
        r = by_id_full[fid]
        entry = by_id_entry[fid]
        render(r, entry, "sample", OUT_DIR)
    for r in correct_pick:
        entry = by_id_entry[r["id"]]
        render(r, entry, "correct", OUT_DIR)

    print(f"Saved -> {OUT_DIR}/sample/ ({len(sample_ids)} images)")
    print(f"Saved -> {OUT_DIR}/correct/ ({len(correct_pick)} images)")

    # ---- worst-SSIM cases (from analyze_singles_final.py's section 9) ----
    if os.path.exists(WORST_CASES_PATH):
        worst_cases = json.load(open(WORST_CASES_PATH))
        print(f"singles worst-SSIM cases: {len(worst_cases)}")
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
