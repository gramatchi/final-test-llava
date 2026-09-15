"""Stats for the singles category (n=1, exactly one distortion present).

Covers recall, method accuracy, value accuracy (on a unified severity scale
so jpeg/noise/gamma are comparable), lenient scoring, method memorization,
and a real SSIM-impact comparison (apply the model's predicted fix vs the
best achievable one, on the actual corrupted image).

Outputs: singles_stats.json, singles_report.md, singles_flagged_examples.json.
"""
import json
import re
import os
import sys
import collections
import statistics

import cv2
from skimage.metrics import structural_similarity as ssim

_REPO = "/nfsd/lttm4/tesisti/gramatchi"
sys.path.insert(0, os.path.join(_REPO, "v6"))
from severity_levels import JPEG_QUALITY, NOISE_SIGMA, GAMMA
from methods_jpeg import METHODS as JPEG_METHODS
from methods_noise import METHODS as NOISE_METHODS
from methods_gamma import METHODS as GAMMA_METHODS

sys.path.insert(0, _REPO)
from corruptors import add_jpeg_compression, add_gaussian_noise, add_gamma_corruption

ANSWERS_PATH = os.path.join(_REPO, "final_test/eval_results/final_test_pipeline_answers.jsonl")
TEST_MANIFEST_PATH = os.path.join(_REPO, "final_test/dataset/llava_final_test.json")
RAW_DIR = "/home/gramatchin/data/raw"
OUT_DIR = os.path.join(_REPO, "final_test/stats/results")
os.makedirs(OUT_DIR, exist_ok=True)

MILD_THRESHOLD = 1
SKIP_SSIM = os.environ.get("SKIP_SSIM", "0") == "1"

JPEG_ORDER = sorted(JPEG_QUALITY, reverse=True)
NOISE_ORDER = sorted(NOISE_SIGMA)
METHOD_FN = {
    "jpeg": {name: fn for name, fn in JPEG_METHODS},
    "noise": {name: fn for name, fn in NOISE_METHODS},
    "gamma": {name: fn for name, fn in GAMMA_METHODS},
}
CORRUPT_FN = {
    "jpeg": lambda img, v: add_jpeg_compression(img, quality=v)[0],
    "noise": lambda img, v: add_gaussian_noise(img, sigma=v)[0],
    "gamma": lambda img, v: add_gamma_corruption(img, gamma=v)[0],
}


def gamma_idx(g):
    # darkening/brightening are mirror-image 5-value ladders; convert a
    # brightening ratio to its dark-side equivalent so both compare fairly,
    # then find the closest rung. Direction doesn't matter, only magnitude.
    darks = sorted(x for x in GAMMA if x > 1.0)
    brights = sorted((1.0 / x for x in GAMMA if x < 1.0))
    side = darks if g > 1.0 else brights
    ratio = g if g > 1.0 else 1.0 / g
    return min(range(len(side)), key=lambda i: abs(side[i] - ratio))


def sev_idx(t, v):
    # converts a raw param into a common 0=mildest..N=strongest position so
    # jpeg/noise/gamma can be compared fairly. jpeg/noise get 10 levels,
    # gamma only 5 -- don't stretch one onto the other without thinking it through.
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


def claimed_value(output, t):
    # pred_value is only set for true types, so for a hallucinated type we
    # have to pull whatever number the model mentioned out of the raw text
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


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


BATCH_SUBDIR = "extra_images/0004000"


def _relative_path_for(bare_name):
    return f"{BATCH_SUBDIR}/{bare_name}"


def load_ssim_ceiling_lookup(dtype, value_field):
    # best achievable SSIM per photo/severity -- the ceiling section 9 compares against
    raw = json.load(open(os.path.join(_REPO, f"final_test/dataset/{dtype}_per_image/per_image.json")))
    out = {}
    for entry in raw:
        path = _relative_path_for(entry["image"])
        out[path] = {v: d["best_ssim"] for v, d in entry[f"by_{value_field}"].items()}
    return out


def main():
    rows = [json.loads(l) for l in open(ANSWERS_PATH)]
    n1 = [r for r in rows if len(r["true_types"]) == 1]

    test_manifest = json.load(open(TEST_MANIFEST_PATH))
    base_to_image = {e["base_id"]: e["image"] for e in test_manifest}

    stats = {}
    report = []

    def w(line=""):
        report.append(line)

    w("# n=1 (exactly one distortion present)")
    w()

    # ---- 1. overview ----
    # how many images and which distortions
    by_photo_all = collections.defaultdict(list)
    for r in n1:
        base = r["id"].split("_final_")[0]
        by_photo_all[base].append(r)
    stats["n_total"] = len(n1)
    stats["n_unique_photos"] = len(by_photo_all)
    w("## 1. Overview")
    w(f"- total examples: {len(n1)}")
    w(f"- unique photos: {len(by_photo_all)}")
    for t in ["jpeg", "noise", "gamma"]:
        n_t = sum(1 for r in n1 if r["true_types"] == [t])
        w(f"- true type = {t}: {n_t}")
    w()

    # ---- 2. per-type recall (detection) ----
    # if found distortion 
    w("## 2. Recall per type (correctly includes the true type)")
    recall = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if r["true_types"] == [t]]
        rec = [t in r["pred_types"] for r in sub]
        recall[t] = {"n": len(sub), "recall": rate(rec)}
        w(f"- {t}: recall={rate(rec):.3f} (n={len(sub)})")
    stats["recall_per_type"] = recall
    w()

    # ---- 3. error breakdown ----
    # two types of errors (missed - over)
    w("## 3. Error breakdown (when true type is NOT correctly/exactly matched)")
    wrong = [r for r in n1 if not r["type_set_correct"]]
    #missed
    missed_entirely = [r for r in wrong if r["true_types"][0] not in r["pred_types"]]
    #over
    over_detected = [r for r in wrong if r["true_types"][0] in r["pred_types"] and len(r["pred_types"]) > 1]
    stats["n_wrong"] = len(wrong)
    stats["missed_entirely"] = len(missed_entirely)
    stats["over_detected"] = len(over_detected)
    w(f"- wrong: {len(wrong)}/{len(n1)} ({100*len(wrong)/len(n1):.1f}%)")
    w(f"  - true type missed entirely (not in pred at all): {len(missed_entirely)} ({100*len(missed_entirely)/len(wrong):.1f}% of wrong)")
    w(f"  - true type found, but extra type(s) also hallucinated: {len(over_detected)} ({100*len(over_detected)/len(wrong):.1f}% of wrong)")
    w()
    
    #which one extra
    w("### 3b. Which extra type gets hallucinated (over-detection cases, exactly 1 extra)")
    over_one_extra = [r for r in over_detected if len(r["pred_types"]) == 2]
    extra_type_dist = collections.Counter()
    for r in over_one_extra:
        extra = [p for p in r["pred_types"] if p != r["true_types"][0]][0]
        extra_type_dist[extra] += 1
    stats["over_detection_extra_type"] = dict(extra_type_dist)
    for t, c in extra_type_dist.most_common():
        w(f"  - extra={t}: {c} ({100*c/len(over_one_extra):.1f}%)" if over_one_extra else "")
    w()
     #if extra why maybe parameter is too high
    w("### 3c. Over-detection rate by the TRUE distortion's OWN severity level")
    w("(does a stronger real distortion make the model more likely to ALSO hallucinate an")
    w("extra, phantom type on top of it? tests the opposite end of the mild-severity story:")
    w("mild severity -> more MISSES of the real type; strong severity -> more EXTRA")
    w("hallucinated types, checked here.)")
    over_detection_by_severity = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if r["true_types"] == [t]]
        by_sev = collections.defaultdict(list)
        for r in sub:
            idx = sev_idx(t, r["steps"][t]["true_value"])
            by_sev[idx].append(len(r["pred_types"]) > 1)
        max_idx = 5 if t == "gamma" else 10
        over_detection_by_severity[t] = {
            str(i): {"rate": rate(by_sev[i]), "n": len(by_sev[i])} for i in range(max_idx) if i in by_sev
        }
        w(f"- {t}:")
        for i in range(max_idx):
            if i in by_sev:
                w(f"    sev_idx={i}: over-detection rate={rate(by_sev[i]):.3f} (n={len(by_sev[i])})")
    stats["over_detection_by_true_severity"] = over_detection_by_severity
    w()

    # ---- 3d. type confusion: not just "missed", but "misread as a different type" ----
    # Section 3's "missed_entirely" bucket lumps together two very different
    # failures: (a) the model said "no distortion" (blind to the artifact
    # entirely) vs (b) the model confidently named a DIFFERENT single type
    # instead of the true one (it saw *something* wrong, just misdiagnosed
    # which kind -- e.g. true=jpeg but it says noise). (b) is a more specific,
    # more interesting error mode for the report, so split it out here and
    # build a true-type -> predicted-type confusion table.
    w("### 3d. Type confusion (true type missed AND a different type predicted in its place)")
    w("(splits the 'missed entirely' bucket from section 3 into two qualitatively different")
    w("failures: SILENT miss -- the model says the photo is clean -- vs TYPE CONFUSION -- the")
    w("model is confident something is wrong, it just names the wrong distortion family.)")
    silent_miss = [r for r in missed_entirely if len(r["pred_types"]) == 0]
    type_confusion = [r for r in missed_entirely if len(r["pred_types"]) > 0]
    stats["missed_silent"] = len(silent_miss)
    stats["missed_type_confusion"] = len(type_confusion)
    if missed_entirely:
        w(f"- of {len(missed_entirely)} missed-entirely cases:")
        w(f"  - silent miss (predicted no distortion at all): {len(silent_miss)} "
          f"({100*len(silent_miss)/len(missed_entirely):.1f}%)")
        w(f"  - type confusion (named a different type instead): {len(type_confusion)} "
          f"({100*len(type_confusion)/len(missed_entirely):.1f}%)")
    w()
    w("  confusion matrix (true type -> what was predicted instead, type-confusion cases only):")
    confusion_matrix = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in type_confusion if r["true_types"][0] == t]
        if not sub:
            continue
        pred_dist = collections.Counter(tuple(sorted(r["pred_types"])) for r in sub)
        confusion_matrix[t] = {
            "n_confused": len(sub),
            "predicted_instead": {"/".join(k) if k else "(none)": v for k, v in pred_dist.items()},
        }
        w(f"    true={t} ({len(sub)} confused examples):")
        for pred_combo, cnt in pred_dist.most_common():
            w(f"      -> predicted {'/'.join(pred_combo)}: {cnt} ({100*cnt/len(sub):.1f}%)")
    stats["type_confusion_matrix"] = confusion_matrix
    w()

    # ---- 3e. silent-miss rate by severity level ----
    # Direct continuation of 3d's silent-vs-confused split, now broken down
    # by the true distortion's OWN severity: at which severity levels does
    # the model outright say the photo is clean (as opposed to section 4's
    # plain recall, whose complement mixes silent misses with type-confusion
    # misses together)?
    w("### 3e. Silent-miss rate by severity level (model said \"no distortion\" entirely)")
    w("(narrower than 1-recall in section 4, which also counts type-confusion misses --")
    w("this counts ONLY pred_types == [] cases, i.e. the model saw nothing wrong at all)")
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if r["true_types"] == [t]]
        by_sev_silent = collections.defaultdict(list)
        for r in sub:
            idx = sev_idx(t, r["steps"][t]["true_value"])
            by_sev_silent[idx].append(len(r["pred_types"]) == 0)
        w(f"- {t}:")
        max_idx = 5 if t == "gamma" else 10
        for i in range(max_idx):
            if i in by_sev_silent:
                w(f"    idx={i}: silent_miss_rate={rate(by_sev_silent[i]):.3f} (n={len(by_sev_silent[i])})")
        stats.setdefault("silent_miss_rate_by_severity", {})[t] = {
            str(i): {"rate": rate(v), "n": len(v)} for i, v in sorted(by_sev_silent.items())
        }
    w()

    # ---- 4. recall/value_accuracy by severity level ----
    w("## 4. Recall and value_accuracy by severity level (0=mildest)")
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if r["true_types"] == [t]]
        by_sev_recall = collections.defaultdict(list)
        by_sev_value = collections.defaultdict(list)
        for r in sub:
            idx = sev_idx(t, r["steps"][t]["true_value"])
            by_sev_recall[idx].append(t in r["pred_types"])
            by_sev_value[idx].append(r["steps"][t]["value_correct"])
        w(f"- {t}:")
        max_idx = 5 if t == "gamma" else 10
        for i in range(max_idx):
            if i in by_sev_recall:
                w(f"    idx={i}: recall={rate(by_sev_recall[i]):.3f}, value_acc={rate(by_sev_value[i]):.3f} (n={len(by_sev_recall[i])})")
    w()

    # ---- 5. method accuracy ----
    w("## 5. Method accuracy per type")
    method_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if t in r["steps"]]
        ma = rate([r["steps"][t]["method_correct"] for r in sub])
        mfa = rate([r["steps"][t]["method_family_correct"] for r in sub])
        method_stats[t] = {"n": len(sub), "method_accuracy": ma, "method_family_accuracy": mfa}
        w(f"- {t}: method_accuracy={ma:.3f}, method_family_accuracy={mfa:.3f} (n={len(sub)})")
    stats["method_accuracy"] = method_stats
    w()

    w("### 5b. Method accuracy GIVEN the type-set is cleanly correct (no drop, no extra hallucination)")
    w("(order is trivial for a single distortion, so this is the singles analog of the")
    w("pairs/triples 'given type+order correct' full-pipeline-correct endpoint)")
    type_correct = [r for r in n1 if r["type_set_correct"]]
    w(f"- n with clean type_set_correct: {len(type_correct)}/{len(n1)} ({100*len(type_correct)/len(n1):.1f}%)")
    method_given_correct = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in type_correct if t in r["steps"]]
        ma = rate([r["steps"][t]["method_correct"] for r in sub])
        mfa = rate([r["steps"][t]["method_family_correct"] for r in sub])
        method_given_correct[t] = {"n": len(sub), "method_accuracy": ma, "method_family_accuracy": mfa}
        w(f"- {t}: method_accuracy={ma:.3f}, method_family_accuracy={mfa:.3f} (n={len(sub)})")
    stats["method_accuracy_given_type_correct"] = method_given_correct
    w()

    # ---- 6. value accuracy: unified severity-index definition ----
    w("## 6. Value accuracy (unified severity-index definition:")
    w("all three types are judged on the same 10-level yardstick -- exact index match,")
    w("or within one adjacent level.)")
    value_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if t in r["steps"]]
        exact = rate([r["steps"][t]["value_correct"] for r in sub])
        within1 = rate([r["steps"][t]["value_correct_within1"] for r in sub])
        dists = []
        for r in sub:
            pv = r["steps"][t]["pred_value"]
            if pv is None:
                continue
            dists.append(abs(sev_idx(t, pv) - sev_idx(t, r["steps"][t]["true_value"])))
        dist_counter = collections.Counter(dists)
        value_stats[t] = {
            "n": len(sub), "exact_fraction": exact, "within1_fraction": within1,
            "index_distance_distribution": dict(sorted(dist_counter.items())),
        }
        w(f"- {t}: exact={exact:.3f}, within-1-level={within1:.3f} (n={len(sub)})")
    stats["value_accuracy"] = value_stats
    w()

    # ---- 6b. value accuracy excluding the weakest severity level(s) ----
    w("## 6b. Value accuracy excluding the weakest severity level(s)")
    w("(drops examples at/below the threshold entirely from both numerator and")
    w("denominator -- a missed detection is NOT reinterpreted as a mild-level guess,")
    w("it is simply excluded here if its true severity is at/below the threshold)")
    excl_value_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub_all = [r for r in n1 if t in r["steps"]]
        excl_value_stats[t] = {}
        w(f"- {t}:")
        exact_all = rate([r["steps"][t]["value_correct"] for r in sub_all])
        within1_all = rate([r["steps"][t]["value_correct_within1"] for r in sub_all])
        w(f"    all severities: exact={exact_all:.3f}, within-1-level={within1_all:.3f} (n={len(sub_all)})")
        excl_value_stats[t]["all"] = {"n": len(sub_all), "exact_fraction": exact_all, "within1_fraction": within1_all}
        for thresh in [0, 1]:
            sub = [r for r in sub_all if sev_idx(t, r["steps"][t]["true_value"]) > thresh]
            exact = rate([r["steps"][t]["value_correct"] for r in sub])
            within1 = rate([r["steps"][t]["value_correct_within1"] for r in sub])
            w(f"    excl sev<={thresh}: exact={exact:.3f}, within-1-level={within1:.3f} (n={len(sub)})")
            excl_value_stats[t][f"excl_sev<={thresh}"] = {"n": len(sub), "exact_fraction": exact, "within1_fraction": within1}
    stats["value_accuracy_excluding_weak_severity"] = excl_value_stats
    w()

    # ---- 6c. within-1/within-2 value accuracy, "graduated miss" variant ----
    w("## 6c. Within-1 / within-2 value accuracy: strict miss vs graduated miss")
    w("(strict: a missed detection is always counted wrong, regardless of true")
    w("severity -- same numbers as section 6/6b. graduated: a missed detection is")
    w("treated as an implicit \"index -1\" guess, one step milder than the mildest")
    w("real level, so missing a very mild true distortion counts as a small")
    w("distance-1 error instead of an unqualified failure.)")
    graduated_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        sub = [r for r in n1 if t in r["steps"]]
        n = len(sub)
        exact = rate([r["steps"][t]["value_correct"] for r in sub])
        w1_strict = w2_strict = w1_grad = w2_grad = 0
        for r in sub:
            pv = r["steps"][t]["pred_value"]
            true_idx = sev_idx(t, r["steps"][t]["true_value"])
            if pv is not None:
                d = abs(true_idx - sev_idx(t, pv))
                w1_strict += d <= 1
                w2_strict += d <= 2
            pred_idx_g = -1 if pv is None else sev_idx(t, pv)
            d_g = abs(true_idx - pred_idx_g)
            w1_grad += d_g <= 1
            w2_grad += d_g <= 2
        graduated_stats[t] = {
            "n": n, "exact_fraction": exact,
            "within1_strict": w1_strict / n, "within1_graduated": w1_grad / n,
            "within2_strict": w2_strict / n, "within2_graduated": w2_grad / n,
        }
        w(f"- {t}: exact={exact:.3f}  "
          f"within1(strict/graduated)={w1_strict/n:.3f}/{w1_grad/n:.3f}  "
          f"within2(strict/graduated)={w2_strict/n:.3f}/{w2_grad/n:.3f}  (n={n})")
    stats["value_accuracy_graduated_miss"] = graduated_stats
    w()

    # ---- 7. lenient type_set_accuracy (forgive one extra mild hallucinated type) ----
    w("## 7. Lenient type_set_accuracy (forgive one mild add OR drop)")
    w("(\"extra-only\" forgives ONLY the over-detection case -- true type found, one extra")
    w("hallucinated at mild severity. \"symmetric\" ALSO forgives the opposite failure: the")
    w("true type dropped entirely, judged by its own true severity since nothing was claimed.)")

    def lenient_correct_extra_only(r, mild_thresh):
        true_t, pred_t = set(r["true_types"]), set(r["pred_types"])
        if true_t == pred_t:
            return True
        sym_diff = true_t.symmetric_difference(pred_t)
        if len(sym_diff) != 1:
            return False
        extra = next(iter(sym_diff))
        if extra in pred_t and extra not in true_t:
            v = claimed_value(r["output"], extra)
            if v is None:
                return False
            return sev_idx(extra, v) <= mild_thresh
        return False

    def lenient_correct_symmetric(r, mild_thresh):
        # same as above but also forgives a dropped type if its true
        # severity was mild (nothing was claimed for it, so we go by ground truth)
        true_t, pred_t = set(r["true_types"]), set(r["pred_types"])
        if true_t == pred_t:
            return True
        sym_diff = true_t.symmetric_difference(pred_t)
        if len(sym_diff) != 1:
            return False
        t = next(iter(sym_diff))
        if t in pred_t and t not in true_t:
            v = claimed_value(r["output"], t)
            if v is None:
                return False
            return sev_idx(t, v) <= mild_thresh
        if t in true_t and t not in pred_t:
            return sev_idx(t, r["steps"][t]["true_value"]) <= mild_thresh
        return False

    strict_acc = rate([r["type_set_correct"] for r in n1])
    lenient_stats = {}
    for thresh, label in [(0, "sev<=0"), (1, "sev<=1")]:
        extra_only = rate([lenient_correct_extra_only(r, thresh) for r in n1])
        symmetric = rate([lenient_correct_symmetric(r, thresh) for r in n1])
        lenient_stats[label] = {"extra_only": extra_only, "symmetric": symmetric}
        w(f"- strict={strict_acc:.3f} -> lenient ({label}): extra_only={extra_only:.3f}, symmetric={symmetric:.3f}")
    stats["strict_type_set_accuracy"] = strict_acc
    stats["lenient_type_set_accuracy"] = lenient_stats
    w()

    # ---- 8. method "memorization" check ----
    w("## 8. Method memorization check (true frequency vs predicted frequency)")
    memorization = {}
    for t in ["jpeg", "noise"]:
        sub = [r for r in n1 if t in r["steps"]]
        true_c = collections.Counter(r["steps"][t]["true_method"] for r in sub)
        pred_c = collections.Counter(r["steps"][t]["pred_method"] for r in sub)
        n = len(sub)
        methods = sorted(set(true_c) | set(pred_c), key=lambda m: -pred_c.get(m, 0))
        memorization[t] = {
            m: {"true_pct": 100 * true_c.get(m, 0) / n, "pred_pct": 100 * pred_c.get(m, 0) / n}
            for m in methods
        }
        w(f"- {t}:")
        for m in methods:
            tp, pp = memorization[t][m]["true_pct"], memorization[t][m]["pred_pct"]
            flag = " <- over-used" if pp - tp > 5 else ""
            w(f"    {m}: true={tp:.1f}%  pred={pp:.1f}%{flag}")
    stats["method_memorization"] = memorization
    w()

    # ---- 9. SSIM impact: predicted (method, value) vs best achievable ----
    # gamma's ceiling is computed on the fly (restore with 'lut' at the true
    # value -- it's the exact inverse, nothing to search for). Examples with
    # no usable prediction count as "no restoration applied" rather than
    # being dropped, since that's what a deployed system would actually do.
    w("## 9. SSIM impact of prediction errors (jpeg/noise/gamma -- real method choice)")
    w("For each example: corrupt the real photo at the TRUE severity, restore using the")
    w("model's PREDICTED method+value (what a deployed system would actually do not knowing")
    w("ground truth), measure SSIM vs the clean original. Compare to the best achievable SSIM")
    w("(best method at the true severity). If the model gave no usable prediction (type missed,")
    w("or method/value unparseable), 'achieved' = the UNRESTORED corrupted image -- this case is")
    w("counted in the overall average, not dropped, since that's what a real deployed system")
    w("would actually leave the user with.")
    ssim_impact = {}
    ssim_types = [] if SKIP_SSIM else ["jpeg", "noise", "gamma"]
    all_ssim_cases = []  # every example across all 3 types, with full context, for the worst-cases file
    if SKIP_SSIM:
        w("(SKIPPED -- SKIP_SSIM=1)")
    for t in ssim_types:
        ceiling_lookup = None
        if t != "gamma":
            value_field = "quality" if t == "jpeg" else "sigma"
            ceiling_lookup = load_ssim_ceiling_lookup(t, value_field)
        sub = [r for r in n1 if t in r["steps"]]
        deltas_compared, deltas_no_restoration, all_deltas = [], [], []
        n_truly_skipped = 0
        for r in sub:
            base = r["id"].split("_final_")[0]
            img_rel = base_to_image.get(base)
            if img_rel is None:
                n_truly_skipped += 1
                continue
            img = cv2.imread(os.path.join(RAW_DIR, img_rel))
            if img is None:
                n_truly_skipped += 1
                continue
            true_v = r["steps"][t]["true_value"]
            corrupted = CORRUPT_FN[t](img, true_v)

            if t == "gamma":
                ceiling_ssim = _ssim(img, METHOD_FN["gamma"]["lut"](corrupted, true_v))
            else:
                ceiling = ceiling_lookup.get(img_rel, {}).get(str(int(true_v)) if t == "jpeg" else str(true_v))
                if ceiling is None:
                    n_truly_skipped += 1
                    continue
                ceiling_ssim = ceiling["ssim"]

            pred_method = r["steps"][t]["pred_method"]
            pred_v = r["steps"][t]["pred_value"]
            has_prediction = pred_method in METHOD_FN[t] and pred_v is not None and not (t == "gamma" and pred_v <= 0)
            if has_prediction:
                try:
                    restored = METHOD_FN[t][pred_method](corrupted, pred_v)
                    achieved_ssim = _ssim(img, restored)
                except Exception:
                    has_prediction = False
            if not has_prediction:
                achieved_ssim = _ssim(img, corrupted)  # no restoration applied -- deployed system leaves it as-is

            delta = ceiling_ssim - achieved_ssim
            all_deltas.append(delta)
            (deltas_compared if has_prediction else deltas_no_restoration).append(delta)
            all_ssim_cases.append({
                "id": r["id"], "base_id": base, "type": t,
                "image_path": os.path.join(RAW_DIR, img_rel), "image_path_relative": img_rel,
                "true_value": true_v, "pred_method": pred_method, "pred_value": pred_v,
                "had_usable_prediction": has_prediction,
                "ceiling_ssim": ceiling_ssim, "achieved_ssim": achieved_ssim, "ssim_loss": delta,
                "output": r["output"],
            })

        def _stats(lst):
            if not lst:
                return {"n": 0, "mean_ssim_loss": None, "median_ssim_loss": None, "fraction_negligible_loss_lt_0.01": None}
            return {
                "n": len(lst), "mean_ssim_loss": statistics.mean(lst), "median_ssim_loss": statistics.median(lst),
                "fraction_negligible_loss_lt_0.01": sum(1 for d in lst if d < 0.01) / len(lst),
            }

        ssim_impact[t] = {
            "overall": _stats(all_deltas), "compared": _stats(deltas_compared),
            "no_restoration": _stats(deltas_no_restoration), "n_truly_unsimulable": n_truly_skipped,
        }
        w(f"- {t}: n={len(all_deltas)} (of which {len(deltas_compared)} had a usable prediction, "
          f"{len(deltas_no_restoration)} got no restoration applied; {n_truly_skipped} truly unsimulable)")
        if all_deltas:
            w(f"    OVERALL (all {len(all_deltas)}): mean SSIM lost vs ideal: {statistics.mean(all_deltas):.4f}, "
              f"median: {statistics.median(all_deltas):.4f}, "
              f"fraction negligible (<0.01): {ssim_impact[t]['overall']['fraction_negligible_loss_lt_0.01']:.3f}")
        if deltas_compared:
            w(f"    -- with prediction ({len(deltas_compared)}): mean lost {statistics.mean(deltas_compared):.4f}, "
              f"median {statistics.median(deltas_compared):.4f}")
        if deltas_no_restoration:
            w(f"    -- no restoration applied ({len(deltas_no_restoration)}): mean lost {statistics.mean(deltas_no_restoration):.4f}, "
              f"median {statistics.median(deltas_no_restoration):.4f}")
    stats["ssim_impact"] = ssim_impact

    # save the 10 worst cases (by SSIM lost) across all 3 types, with full context
    if all_ssim_cases:
        worst_ssim_cases = sorted(all_ssim_cases, key=lambda c: -c["ssim_loss"])[:10]
        worst_cases_dir = os.path.join(_REPO, "final_test/stats/worst_cases")
        os.makedirs(worst_cases_dir, exist_ok=True)
        with open(os.path.join(worst_cases_dir, "singles_ssim_worst_cases.json"), "w") as f:
            json.dump(worst_ssim_cases, f, indent=2)
        w(f"- top 10 worst individual cases (by SSIM lost, across all 3 types) saved -> "
          f"stats/worst_cases/singles_ssim_worst_cases.json")
    w()

    # ---- 10. does accuracy depend on which question phrasing was used? ----
    # The dataset asks about each photo with one of ~20 different phrasings of
    # the same request (build_dataset_manifest_final.py's QUESTIONS list),
    # picked at random per example, independent of the actual distortion. If
    # the model were purely reading the image, type_set_accuracy should be
    # roughly flat across phrasings; a systematic spread would mean the model
    # is (at least partly) sensitive to HOW the question is worded, not just
    # to WHAT is actually in the photo.
    w("## 10. Does accuracy depend on the question phrasing?")
    w("(~20 different phrasings of the same request are used, chosen at random per example --")
    w("independent of the actual distortion. If wording doesn't matter, type_set_accuracy should")
    w("be roughly flat across all of them. NOTE: with ~1000 examples over ~20 phrasings, n per")
    w("phrasing is small (~50) -- treat modest spread as noise, only a large spread as a real signal.)")
    id_to_question = {e["id"]: e["conversations"][0]["value"].replace("<image>", "").strip() for e in test_manifest}
    by_question = collections.defaultdict(list)
    for r in n1:
        by_question[id_to_question.get(r["id"], "(unknown)")].append(r["type_set_correct"])
    q_stats = {q: {"n": len(v), "type_set_accuracy": rate(v)} for q, v in by_question.items()}
    accs = [s["type_set_accuracy"] for s in q_stats.values()]
    w(f"- {len(q_stats)} distinct phrasings seen, ~{len(n1)//len(q_stats)} examples each on average")
    if len(accs) > 1:
        w(f"- type_set_accuracy across phrasings: min={min(accs):.3f}, max={max(accs):.3f}, "
          f"spread={max(accs)-min(accs):.3f}, stdev={statistics.stdev(accs):.3f}")
    w("- per-phrasing breakdown, worst to best:")
    for q, s in sorted(q_stats.items(), key=lambda kv: kv[1]["type_set_accuracy"]):
        short_q = q if len(q) <= 70 else q[:67] + "..."
        w(f"    {s['type_set_accuracy']:.3f} (n={s['n']}): \"{short_q}\"")
    stats["accuracy_by_question_phrasing"] = q_stats
    w()

    # ---- 11. flagged examples (miss / over-detection cases) ----
    flagged = []
    for r in wrong:
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        flagged.append({
            "id": r["id"], "base_id": base,
            "image_path": os.path.join(RAW_DIR, img_rel) if img_rel else None,
            "image_path_relative": img_rel,
            "true_types": r["true_types"], "pred_types": r["pred_types"],
            "output": r["output"],
        })
    stats["n_flagged_examples"] = len(flagged)
    w("## 11. Flagged examples")
    w(f"- {len(flagged)} wrong examples saved -> singles_flagged_examples.json")
    w()

    with open(os.path.join(OUT_DIR, "singles_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "singles_report.md"), "w") as f:
        f.write("\n".join(report))
    with open(os.path.join(OUT_DIR, "singles_flagged_examples.json"), "w") as f:
        json.dump(flagged, f, indent=2)

    print("\n".join(report))
    print(f"\nSaved -> {OUT_DIR}/singles_stats.json")
    print(f"Saved -> {OUT_DIR}/singles_report.md")
    print(f"Saved -> {OUT_DIR}/singles_flagged_examples.json")


if __name__ == "__main__":
    main()
