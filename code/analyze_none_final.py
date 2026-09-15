"""Stats for the "none" category (n=0, no real distortion present).

Breaks down how well the model recognizes a clean photo as clean.
"""
import json
import re
import collections
import statistics
import sys
import os

import cv2
from skimage.metrics import structural_similarity as ssim

_REPO = "/nfsd/lttm4/tesisti/gramatchi"
sys.path.insert(0, os.path.join(_REPO, "v6"))
from severity_levels import JPEG_QUALITY, NOISE_SIGMA, GAMMA
from methods_jpeg import METHODS as JPEG_METHODS
from methods_noise import METHODS as NOISE_METHODS
from methods_gamma import METHODS as GAMMA_METHODS

ANSWERS_PATH = os.path.join(_REPO, "final_test/eval_results/final_test_pipeline_answers.jsonl")
TEST_MANIFEST_PATH = os.path.join(_REPO, "final_test/dataset/llava_final_test.json")
RAW_DIR = "/home/gramatchin/data/raw"
OUT_DIR = os.path.join(_REPO, "final_test/stats/results")
os.makedirs(OUT_DIR, exist_ok=True)

MILD_THRESHOLD = 1  # severity index <= this counts as "mild" (indices are 0..9, 0=weakest)

JPEG_ORDER = sorted(JPEG_QUALITY, reverse=True)   # index 0 = mildest (highest quality)
NOISE_ORDER = sorted(NOISE_SIGMA) # index 0 = mildest (lowest sigma)


def gamma_idx(g):
    # darkening and brightening are mirror-image ladders of 5 values each;
    # convert a brightening ratio to its dark-side equivalent (1/g) so both
    # sides compare on the same footing, then pick the closest rung.
    # Direction is discarded on purpose -- only magnitude matters here.
    darks = sorted(x for x in GAMMA if x > 1.0)
    brights = sorted((1.0 / x for x in GAMMA if x < 1.0))
    side = darks if g > 1.0 else brights
    ratio = g if g > 1.0 else 1.0 / g
    return min(range(len(side)), key=lambda i: abs(side[i] - ratio))


def sev_idx(t, v):
    # maps a raw param (jpeg quality, noise sigma, gamma) onto a common
    # 0=mildest..N=strongest scale so the three types compare fairly
    if t == "jpeg":
        return min(range(len(JPEG_ORDER)), key=lambda i: abs(JPEG_ORDER[i] - v))
    if t == "noise":
        return min(range(len(NOISE_ORDER)), key=lambda i: abs(NOISE_ORDER[i] - v))
    if t == "gamma":
        return gamma_idx(v)
    raise ValueError(t)


QUALITY_RE = re.compile(r"quality\D{0,10}?(\d+)", re.IGNORECASE)
SIGMA_RE = re.compile(r"sigma\D{0,10}?([0-9]*\.?[0-9]+)", re.IGNORECASE)
GAMMA_RE = re.compile(r"gamma\D{0,5}?([0-9]*\.?[0-9]+)", re.IGNORECASE)
VALUE_RE = {"jpeg": QUALITY_RE, "noise": SIGMA_RE, "gamma": GAMMA_RE}

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


def claimed_method(output):
    # since there's no true type here, pred_method (from the eval script)
    # is never populated -- pull the method name straight from the text
    m = METHOD_RE.search(output)
    return m.group(1) if m else None


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


def claimed_value(output, t):
    # same deal as claimed_method -- pred_value isn't set for a hallucinated
    # type, so pull whatever number the model mentioned out of the text
    m = VALUE_RE[t].search(output)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def rate(lst):
    return sum(lst) / len(lst) if lst else None


def image_brightness(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def main():
    rows = [json.loads(l) for l in open(ANSWERS_PATH)]
    n0 = [r for r in rows if len(r["true_types"]) == 0]

    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    base_to_image = {e["base_id"]: e["image"] for e in test_manifest}

    stats = {}
    report = []

    def w(line=""):
        report.append(line)

    w("# n=0 (\"none\" -- no distortions present)")
    w()

    # ---- 1. overview ----
    by_photo_all = collections.defaultdict(list)
    for r in n0:
        base = r["id"].split("_final_")[0]
        by_photo_all[base].append(r)
    n_photos = len(by_photo_all)
    entries_per_photo = len(n0) // n_photos if n_photos else 0
    stats["n_total"] = len(n0)
    stats["n_unique_photos"] = n_photos
    stats["entries_per_photo"] = entries_per_photo
    w("## 1. Overview")
    w(f"- total examples: {len(n0)}")
    w(f"- unique photos: {n_photos}")
    w(f"- entries per photo: {entries_per_photo}")
    w()

    # ---- 2. headline accuracy ----
    correct = [r["type_set_correct"] for r in n0]
    acc = rate(correct)
    stats["accuracy"] = acc
    w("## 2. Headline accuracy")
    w(f"- correctly says \"no distortion\": {acc:.4f} ({sum(correct)}/{len(n0)})")
    w()

    # ---- 3. error breakdown by hallucination size ----
    wrong = [r for r in n0 if not r["type_set_correct"]]
    size_dist = collections.Counter(len(r["pred_types"]) for r in wrong)
    stats["n_wrong"] = len(wrong)
    stats["wrong_by_hallucinated_count"] = dict(sorted(size_dist.items()))
    w("## 3. Error breakdown by hallucination size")
    w(f"- wrong answers: {len(wrong)}/{len(n0)} ({100*len(wrong)/len(n0):.1f}%)")
    for k in sorted(size_dist):
        w(f"  - claims {k} distortion type(s): {size_dist[k]} ({100*size_dist[k]/len(wrong):.1f}% of wrong)")
    w()

    # ---- 4. which type gets hallucinated (single-type errors only) ----
    one_type_wrong = [r for r in wrong if len(r["pred_types"]) == 1]
    type_dist = collections.Counter(r["pred_types"][0] for r in one_type_wrong)
    stats["single_type_hallucination_n"] = len(one_type_wrong)
    stats["single_type_hallucination_by_type"] = {
        t: {"count": c, "fraction": c / len(one_type_wrong)} for t, c in type_dist.items()
    } if one_type_wrong else {}
    w("## 4. Which type gets hallucinated (single-type errors only)")
    w(f"- n={len(one_type_wrong)}")
    for t, c in type_dist.most_common():
        w(f"  - {t}: {c} ({100*c/len(one_type_wrong):.1f}%)")
    w()

    # ---- 5. claimed severity when hallucinating ----
    w("## 5. Claimed severity when hallucinating (0=mildest)")
    severity_by_type = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in one_type_wrong if r["pred_types"][0] == t]
        idxs = []
        for r in sub:
            v = claimed_value(r["output"], t)
            if v is not None:
                idxs.append(sev_idx(t, v))
        if not idxs:
            continue
        dist = collections.Counter(idxs)
        mild_frac = sum(1 for i in idxs if i <= MILD_THRESHOLD) / len(idxs)
        max_idx = 5 if t == "gamma" else 10
        severity_by_type[t] = {
            "n": len(idxs),
            "distribution": {str(i): dist.get(i, 0) for i in range(max_idx)},
            "mild_fraction": mild_frac,
        }
        w(f"- {t} (n={len(idxs)}): " + ", ".join(f"{i}:{dist.get(i,0)}" for i in range(max_idx)))
        w(f"  -> {100*mild_frac:.1f}% claimed at mild severity (index <= {MILD_THRESHOLD})")
    stats["claimed_severity_when_hallucinating"] = severity_by_type
    w()

    # ---- 6. brightness correlation check ----
    w("## 6. Brightness correlation check (does unusual overall brightness predict gamma hallucination?)")
    brightness = {}
    for base, img_rel in base_to_image.items():
        if base not in by_photo_all:
            continue
        path = os.path.join(RAW_DIR, img_rel)
        b = image_brightness(path)
        if b is not None:
            brightness[base] = b

    pop_vals = list(brightness.values())
    pop_mean = statistics.mean(pop_vals)
    pop_std = statistics.pstdev(pop_vals)

    flagged_photos = [r["id"].split("_final_")[0] for r in n0 if "gamma" in r["pred_types"]]
    flagged_z = [abs((brightness[b] - pop_mean) / pop_std) for b in flagged_photos if b in brightness]
    all_z = [abs((v - pop_mean) / pop_std) for v in pop_vals]

    stats["brightness_check"] = {
        "population_mean": pop_mean,
        "population_std": pop_std,
        "n_flagged_photos": len(flagged_photos),
        "mean_abs_zscore_flagged": statistics.mean(flagged_z) if flagged_z else None,
        "mean_abs_zscore_all": statistics.mean(all_z) if all_z else None,
    }
    w(f"- population brightness (grayscale mean pixel value): mean={pop_mean:.2f}, std={pop_std:.2f}")
    w(f"- photos with any gamma false-positive: {len(flagged_photos)}")
    if flagged_z:
        w(f"- mean |z-score| of brightness, flagged photos: {statistics.mean(flagged_z):.3f}")
        w(f"- mean |z-score| of brightness, ALL photos:     {statistics.mean(all_z):.3f}")
        ratio = statistics.mean(flagged_z) / statistics.mean(all_z) if statistics.mean(all_z) else None
        if ratio:
            w(f"- flagged photos are {ratio:.2f}x further from average brightness than a typical photo")
    w()

    # ---- 7. does accuracy depend on which question phrasing was used? ----
    w("## 7. Does accuracy depend on the question phrasing?")
    w("(~20 phrasings used at random, independent of the (here: always clean) ground truth.")
    w("NOTE: n per phrasing is small (~50) -- treat modest spread as noise.)")
    id_to_question = {e["id"]: e["conversations"][0]["value"].replace("<image>", "").strip() for e in test_manifest}
    by_question = collections.defaultdict(list)
    for r in n0:
        by_question[id_to_question.get(r["id"], "(unknown)")].append(r["type_set_correct"])
    q_stats = {q: {"n": len(v), "accuracy": rate(v)} for q, v in by_question.items()}
    accs = [s["accuracy"] for s in q_stats.values()]
    w(f"- {len(q_stats)} distinct phrasings seen, ~{len(n0)//len(q_stats)} examples each on average")
    if len(accs) > 1:
        w(f"- accuracy across phrasings: min={min(accs):.3f}, max={max(accs):.3f}, "
          f"spread={max(accs)-min(accs):.3f}, stdev={statistics.stdev(accs):.3f}")
    w("- per-phrasing breakdown, worst to best:")
    for q, s in sorted(q_stats.items(), key=lambda kv: kv[1]["accuracy"]):
        short_q = q if len(q) <= 70 else q[:67] + "..."
        w(f"    {s['accuracy']:.3f} (n={s['n']}): \"{short_q}\"")
    stats["accuracy_by_question_phrasing"] = q_stats
    w()

    # ---- 8. lenient accuracy (forgive a single mild hallucinated type) ----
    w("## 8. Lenient accuracy (forgive ONE hallucinated type if claimed at mild severity)")
    w("(strict = section 2's exact pred_types==[] requirement. lenient = also count it correct")
    w("if the model named exactly one extra type, as long as its OWN claimed severity for that")
    w("type is mild (index <= threshold). A 2-3 type hallucination is never forgiven here.)")
    lenient_stats = {}

    def lenient_correct(r, mild_thresh):
        if r["type_set_correct"]:
            return True
        if len(r["pred_types"]) != 1:
            return False
        t = r["pred_types"][0]
        v = claimed_value(r["output"], t)
        if v is None:
            return False
        return sev_idx(t, v) <= mild_thresh

    for thresh, label in [(0, "sev<=0"), (1, "sev<=1")]:
        la = rate([lenient_correct(r, thresh) for r in n0])
        lenient_stats[label] = la
        w(f"- strict={acc:.3f} -> lenient ({label})={la:.3f}")
    stats["lenient_accuracy"] = lenient_stats
    w()

    # ---- 9. SSIM impact of hallucinated "fixes" applied to an already-clean photo ----
    # restricted to single-type hallucinations so there's exactly one
    # restoration step to simulate; ceiling is trivially 1.0 since doing
    # nothing to a clean photo is already perfect
    w("## 9. SSIM impact of hallucinated \"fixes\" applied to an already-clean photo")
    w("(restricted to single-type hallucinations so exactly one restoration step is well-defined.")
    w("Ceiling is trivially 1.0 -- doing nothing to a clean photo is already perfect. \"Achieved\"")
    w("applies the model's OWN claimed method+value to the clean original -- what a deployed")
    w("system would actually hand the user if it trusted this false alarm.)")
    ssim_cases = []  # one dict per example, so the worst cases can be saved with full context
    n_skipped_ssim = 0
    for r in one_type_wrong:
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        if img_rel is None:
            n_skipped_ssim += 1
            continue
        img = cv2.imread(os.path.join(RAW_DIR, img_rel))
        if img is None:
            n_skipped_ssim += 1
            continue
        t = r["pred_types"][0]
        pm = claimed_method(r["output"])
        pv = claimed_value(r["output"], t)
        has_prediction = pm in METHOD_FN[t] and pv is not None and not (t == "gamma" and pv <= 0)
        achieved = None
        if has_prediction:
            try:
                restored = METHOD_FN[t][pm](img, pv)
                achieved = _ssim(img, restored)
            except Exception:
                has_prediction = False
        if not has_prediction:
            achieved = 1.0  # no usable prediction -> deployed system applies nothing, stays clean
        ssim_cases.append({
            "id": r["id"], "base_id": base,
            "image_path": os.path.join(RAW_DIR, img_rel), "image_path_relative": img_rel,
            "pred_type": t, "claimed_method": pm, "claimed_value": pv,
            "output": r["output"], "achieved_ssim": achieved, "ssim_loss": 1.0 - achieved,
        })
    ssim_deltas = [c["ssim_loss"] for c in ssim_cases]
    stats["ssim_impact_false_positive_fix"] = {
        "n": len(ssim_deltas), "n_skipped": n_skipped_ssim,
        "mean_ssim_loss": statistics.mean(ssim_deltas) if ssim_deltas else None,
        "median_ssim_loss": statistics.median(ssim_deltas) if ssim_deltas else None,
        "max_ssim_loss": max(ssim_deltas) if ssim_deltas else None,
    }
    # save the 10 worst cases with full context so we don't have to rerun this later
    worst_cases = sorted(ssim_cases, key=lambda c: -c["ssim_loss"])[:10]
    worst_cases_dir = os.path.join(_REPO, "final_test/stats/worst_cases")
    os.makedirs(worst_cases_dir, exist_ok=True)
    with open(os.path.join(worst_cases_dir, "none_ssim_worst_cases.json"), "w") as f:
        json.dump(worst_cases, f, indent=2)
    if ssim_deltas:
        w(f"- n={len(ssim_deltas)} (of {len(one_type_wrong)} single-type hallucinations; "
          f"{n_skipped_ssim} skipped -- image not found)")
        w(f"- mean SSIM lost by \"fixing\" a clean photo:   {statistics.mean(ssim_deltas):.4f}")
        w(f"- median SSIM lost:                            {statistics.median(ssim_deltas):.4f}")
        w(f"- max SSIM lost (worst single case):           {max(ssim_deltas):.4f}")
        w(f"- top 10 worst individual cases saved -> stats/worst_cases/none_ssim_worst_cases.json")
    w()

    # ---- 10. concrete flagged examples (for visual inspection) ----
    flagged_examples = []
    for r in wrong:
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        b = brightness.get(base)
        z = abs((b - pop_mean) / pop_std) if (b is not None) else None
        flagged_examples.append({
            "id": r["id"],
            "base_id": base,
            "image_path": os.path.join(RAW_DIR, img_rel) if img_rel else None,
            "image_path_relative": img_rel,
            "pred_types": r["pred_types"],
            "output": r["output"],
            "brightness": b,
            "brightness_zscore": z,
        })
    stats["n_flagged_examples"] = len(flagged_examples)

    w("## 10. Flagged examples")
    w(f"- {len(flagged_examples)} wrong examples saved with full detail (image path, output text, "
      f"brightness z-score) -> see none_flagged_examples.json")
    w()

    # ---- save everything ----
    with open(os.path.join(OUT_DIR, "none_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "none_report.md"), "w") as f:
        f.write("\n".join(report))
    with open(os.path.join(OUT_DIR, "none_flagged_examples.json"), "w") as f:
        json.dump(flagged_examples, f, indent=2)

    print("\n".join(report))
    print(f"\nSaved -> {OUT_DIR}/none_stats.json")
    print(f"Saved -> {OUT_DIR}/none_report.md")
    print(f"Saved -> {OUT_DIR}/none_flagged_examples.json")


if __name__ == "__main__":
    main()
