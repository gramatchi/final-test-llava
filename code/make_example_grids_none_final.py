"""Renders 3-panel example images for the none category (no real distortion),
focused on false-positive cases -- what would happen to an already-clean
photo if a deployed system trusted the model's hallucinated diagnosis.

Since nothing was actually corrupted, CORRUPTED == ORIGINAL; the interesting
panel is RESTORED, showing the effect of an unnecessary "fix".

Saved under final_test/example_images/none/.
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

ANSWERS_PATH = os.path.join(_REPO, "final_test/eval_results/final_test_pipeline_answers.jsonl")
FLAGGED_PATH = os.path.join(_REPO, "final_test/stats/results/none_flagged_examples.json")
TEST_MANIFEST_PATH = os.path.join(_REPO, "final_test/dataset/llava_final_test.json")
RAW_DIR = "/home/gramatchin/data/raw"
WORST_CASES_PATH = os.path.join(_REPO, "final_test/stats/worst_cases/none_ssim_worst_cases.json")
OUT_DIR = os.path.join(_REPO, "final_test/example_images/none")
os.makedirs(os.path.join(OUT_DIR, "wrong"), exist_ok=True)
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
CAPTION_H = 260
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


def claimed_method(output):
    m = METHOD_RE.search(output)
    return m.group(1) if m else None


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


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
    # orig and corrupted are the same array here (nothing was actually
    # distorted) -- the title makes that explicit so it doesn't look like a bug
    W = PANEL_SIZE * 3 + MARGIN * 4
    H = PANEL_SIZE + CAPTION_H + MARGIN * 2
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    panels = [orig_bgr, corrupted_bgr, restored_bgr]
    titles = ["ORIGINAL (clean)", "\"CORRUPTED\" (= original)", "RESTORED (model's fix applied)"]
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


def main():
    # three galleries: 10 false-positive examples, 10 correctly-classified
    # ones for contrast, and the 10 worst by SSIM loss
    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    by_id_entry = {e["id"]: e for e in test_manifest}

    flagged = json.load(open(FLAGGED_PATH))
    single_halluc = [f for f in flagged if len(f["pred_types"]) == 1]
    print(f"none flagged total: {len(flagged)}, single-type hallucinations: {len(single_halluc)}")

    by_type = {"jpeg": [], "noise": [], "gamma": []}
    for f in single_halluc:
        by_type[f["pred_types"][0]].append(f)

    rng = random.Random(42)
    for t in by_type:
        rng.shuffle(by_type[t])

    # round-robin across types to get good variety, skipping repeated base_ids
    picked = []
    used_bases = set()
    pools = {t: list(lst) for t, lst in by_type.items()}
    types_cycle = ["jpeg", "noise", "gamma"]
    i = 0
    while len(picked) < 10 and any(pools.values()):
        t = types_cycle[i % 3]
        i += 1
        while pools[t]:
            cand = pools[t].pop()
            if cand["base_id"] not in used_bases:
                picked.append(cand)
                used_bases.add(cand["base_id"])
                break

    print(f"picked {len(picked)}: " + ", ".join(f"{p['pred_types'][0]}({p['base_id']})" for p in picked))

    for f in picked:
        img_rel = f["image_path_relative"]
        img = cv2.imread(os.path.join(RAW_DIR, img_rel))
        if img is None:
            print("  skip (image not found):", img_rel)
            continue

        t = f["pred_types"][0]
        pm = claimed_method(f["output"])
        pv = claimed_value(f["output"], t)
        has_prediction = pm in METHOD_FN[t] and pv is not None and not (t == "gamma" and pv <= 0)

        note = ""
        if has_prediction:
            try:
                restored = METHOD_FN[t][pm](img, pv)
            except Exception:
                has_prediction = False
        if not has_prediction:
            restored = img
            note = " (could not parse a usable method/value -- shown unchanged)"

        ssim_restored = _ssim(img, restored)

        entry = by_id_entry.get(f["id"])
        question = entry["conversations"][0]["value"].replace("<image>", "").strip() if entry else "(question not found)"
        left_lines = [
            f'Question (as asked): "{question}"',
            "True state: no distortion applied -- this photo is clean.",
        ]
        center_lines = [f'Model answer (full): "{f["output"]}"']
        right_lines = [
            "SSIM original vs corrupted: 1.000 (identical -- nothing was actually done to the photo)",
            f"SSIM original vs restored: {ssim_restored:.3f}{note}",
            f"  model claimed: {t} via '{pm}' @ {pv}" if pm else "  model's method/value could not be parsed",
        ]

        out_path = os.path.join(OUT_DIR, "wrong", f"{f['id']}.png")
        make_panel_image(img, img, restored, left_lines, center_lines, right_lines, out_path)

    print(f"Saved -> {OUT_DIR}/wrong/ ({len(picked)} images)")

    # ---- 10 correctly-classified examples (model correctly said "no distortion") ----
    rows = [json.loads(l) for l in open(ANSWERS_PATH)]
    n0 = [r for r in rows if len(r["true_types"]) == 0]
    correct_rows = [r for r in n0 if r["type_set_correct"]]
    print(f"none correct total: {len(correct_rows)}")

    rng2 = random.Random(7)
    rng2.shuffle(correct_rows)
    picked_correct = []
    used_bases_c = set()
    for r in correct_rows:
        base = r["id"].split("_final_")[0]
        if base not in used_bases_c:
            picked_correct.append(r)
            used_bases_c.add(base)
        if len(picked_correct) == 10:
            break

    print(f"picked {len(picked_correct)} correct: " + ", ".join(r["id"].split("_final_")[0] for r in picked_correct))

    for r in picked_correct:
        base = r["id"].split("_final_")[0]
        entry = by_id_entry.get(r["id"])
        if entry is None:
            print("  skip (no manifest entry):", r["id"])
            continue
        img_rel = entry["image"]
        img = cv2.imread(os.path.join(RAW_DIR, img_rel))
        if img is None:
            print("  skip (image not found):", img_rel)
            continue

        question = entry["conversations"][0]["value"].replace("<image>", "").strip()
        left_lines = [
            f'Question (as asked): "{question}"',
            "True state: no distortion applied -- this photo is clean.",
        ]
        center_lines = [f'Model answer (full): "{r["output"]}"']
        right_lines = [
            "SSIM original vs corrupted: 1.000 (identical -- nothing was actually done to the photo)",
            "SSIM original vs restored: 1.000 (model correctly applied no fix)",
            "  model correctly said: no distortion present",
        ]

        out_path = os.path.join(OUT_DIR, "correct", f"{r['id']}.png")
        make_panel_image(img, img, img, left_lines, center_lines, right_lines, out_path)

    print(f"Saved -> {OUT_DIR}/correct/ ({len(picked_correct)} images)")

    # ---- worst-SSIM cases (from analyze_none_final.py's section 9 --
    # single-type hallucinations where "fixing" the clean photo hurt SSIM
    # the most) ----
    worst_cases = json.load(open(WORST_CASES_PATH))
    print(f"none worst-SSIM cases: {len(worst_cases)}")

    for c in worst_cases:
        img = cv2.imread(c["image_path"])
        if img is None:
            print("  skip (image not found):", c["image_path"])
            continue

        t, pm, pv = c["pred_type"], c["claimed_method"], c["claimed_value"]
        if pm in METHOD_FN[t] and pv is not None:
            restored = METHOD_FN[t][pm](img, pv)
        else:
            restored = img

        entry = by_id_entry.get(c["id"])
        question = entry["conversations"][0]["value"].replace("<image>", "").strip() if entry else "(question not found)"
        left_lines = [
            f'Question (as asked): "{question}"',
            "True state: no distortion applied -- this photo is clean.",
        ]
        center_lines = [f'Model answer (full): "{c["output"]}"']
        right_lines = [
            "SSIM original vs corrupted: 1.000 (identical -- nothing was actually done to the photo)",
            f"SSIM original vs restored: {c['achieved_ssim']:.3f}  (SSIM lost: {c['ssim_loss']:.3f})",
            f"  model claimed: {t} via '{pm}' @ {pv}",
        ]

        out_path = os.path.join(OUT_DIR, "worst_ssim", f"{c['id']}.png")
        make_panel_image(img, img, restored, left_lines, center_lines, right_lines, out_path)

    print(f"Saved -> {OUT_DIR}/worst_ssim/ ({len(worst_cases)} images)")


if __name__ == "__main__":
    main()
