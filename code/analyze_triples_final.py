"""Stats for the triples category (n=3, all three distortions present).

Only one possible type-set here (jpeg+noise+gamma), so there's no
over-detection scenario -- the focus is order accuracy/bias/per-position
accuracy across the 6 possible restore orders, plus the same method/value/
lenient/memorization/SSIM sections as the other categories.

SKIP_SSIM=1 skips the expensive section (three corrupt+restore steps per
example); run via sbatch for the full pass.

Outputs: triples_stats.json, triples_report.md, triples_flagged_examples.json.
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
    # jpeg/noise/gamma can be compared fairly
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


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


def apply_pipeline(img, order, values_by_type, methods_by_type):
    # used for both the ground-truth ceiling restoration and for replaying
    # the model's own claimed pipeline -- same function, different args
    cur = img
    for t in order:
        fn = METHOD_FN[t][methods_by_type[t]]
        cur = fn(cur, values_by_type[t])
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

    stats = {}
    report = []

    def w(line=""):
        report.append(line)

    w("# n=3 (all three distortions present: jpeg+noise+gamma)")
    w()

    # ---- 1. overview ----
    by_photo_all = collections.defaultdict(list)
    for r in n3:
        base = r["id"].split("_final_")[0]
        by_photo_all[base].append(r)
    stats["n_total"] = len(n3)
    stats["n_unique_photos"] = len(by_photo_all)
    w("## 1. Overview")
    w(f"- total examples: {len(n3)}")
    w(f"- unique photos: {len(by_photo_all)}")
    w()

    # ---- 2. type-set accuracy and confusion ----
    w("## 2. Type-set accuracy and confusion (only one true set: jpeg+noise+gamma)")
    tsa = rate([r["type_set_correct"] for r in n3])
    stats["type_set_accuracy"] = tsa
    pred_dist = collections.Counter(tuple(sorted(r["pred_types"])) for r in n3)
    w(f"- type_set_accuracy: {tsa:.3f}")
    for combo, cnt in pred_dist.most_common():
        tag = "CORRECT" if set(combo) == {"jpeg", "noise", "gamma"} else "wrong"
        w(f"  predicted {'+'.join(combo) if combo else '(nothing)'}: {cnt} ({100*cnt/len(n3):.1f}%) [{tag}]")
    stats["predicted_set_distribution"] = {("+".join(c) if c else "(nothing)"): cnt for c, cnt in pred_dist.items()}
    w()

    w("### 2b. How many of the 3 true types get dropped, when wrong")
    wrong = [r for r in n3 if not r["type_set_correct"]]
    drop_dist = collections.Counter(3 - len(set(r["true_types"]) & set(r["pred_types"])) for r in wrong)
    stats["n_wrong"] = len(wrong)
    stats["dropped_count_distribution"] = dict(sorted(drop_dist.items()))
    for k in sorted(drop_dist):
        w(f"  - {k} type(s) dropped: {drop_dist[k]} ({100*drop_dist[k]/len(wrong):.1f}% of wrong)")
    which_dropped = collections.Counter()
    for r in wrong:
        for t in set(r["true_types"]) - set(r["pred_types"]):
            which_dropped[t] += 1
    stats["which_type_dropped"] = dict(which_dropped)
    w("  - which type(s) get dropped (counting each dropped type once):")
    for t, c in which_dropped.most_common():
        w(f"      {t}: {c}")
    w()

    # ---- 3. order accuracy, bias, confusion, position-wise ----
    w("## 3. Order accuracy, bias, and per-position accuracy")
    type_correct = [r for r in n3 if r["type_set_correct"]]
    order_acc = rate([r["order_correct"] for r in type_correct])
    stats["order_accuracy_given_type_correct"] = order_acc
    w(f"- order_accuracy_given_type_correct: {order_acc:.3f} (n={len(type_correct)})")

    w("### 3a. Accuracy by true order permutation, and most common wrong prediction")
    by_true_order = collections.defaultdict(list)
    for r in type_correct:
        by_true_order[tuple(r["true_order"])].append(r)
    order_by_true = {}
    for order, rs in sorted(by_true_order.items(), key=lambda x: -rate([r["order_correct"] for r in x[1]])):
        acc = rate([r["order_correct"] for r in rs])
        wrong_rs = [r for r in rs if not r["order_correct"]]
        top_wrong = collections.Counter(tuple(r["pred_order"]) for r in wrong_rs).most_common(1)
        top_str = "->".join(top_wrong[0][0]) if top_wrong else None
        order_by_true["->".join(order)] = {"n": len(rs), "accuracy": acc, "most_common_wrong_prediction": top_str}
        w(f"  {'->'.join(order)}: n={len(rs)}, acc={acc:.3f}" + (f", most common wrong -> {top_str}" if top_str else ""))
    stats["order_accuracy_by_true_permutation"] = order_by_true
    w()

    w("### 3b. Predicted-order bias (true frequency should be ~16.7% each)")
    true_dist = collections.Counter(tuple(r["true_order"]) for r in type_correct)
    pred_dist_o = collections.Counter(tuple(r["pred_order"]) for r in type_correct)
    all_orders = sorted(set(true_dist) | set(pred_dist_o), key=lambda o: -pred_dist_o.get(o, 0))
    order_bias = {}
    for o in all_orders:
        key = "->".join(o)
        order_bias[key] = {"true_pct": 100 * true_dist.get(o, 0) / len(type_correct),
                            "pred_pct": 100 * pred_dist_o.get(o, 0) / len(type_correct)}
        w(f"  {key}: true={order_bias[key]['true_pct']:.1f}%  pred={order_bias[key]['pred_pct']:.1f}%")
    stats["order_bias"] = order_bias
    w()

    w("### 3c. Position-wise accuracy (is each type correctly placed in position 1/2/3)")
    pos_acc = collections.defaultdict(list)
    for r in type_correct:
        for i, t in enumerate(r["true_order"]):
            pred_i = r["pred_order"][i] if i < len(r["pred_order"]) else None
            pos_acc[(i, t)].append(pred_i == t)
    position_accuracy = {}
    for (i, t), v in sorted(pos_acc.items()):
        position_accuracy[f"pos{i+1}_{t}"] = {"n": len(v), "accuracy": rate(v)}
        w(f"  position {i+1}, type={t}: n={len(v)}, accuracy={rate(v):.3f}")
    stats["position_accuracy"] = position_accuracy
    w()

    w("### 3d. Error shape: single adjacent swap vs fuller scramble")
    def is_adjacent_swap(true_o, pred_o):
        # true if the predicted order differs from the true one by swapping
        # exactly two adjacent positions, rather than a fuller scramble
        if sorted(true_o) != sorted(pred_o) or len(true_o) != 3:
            return False
        diffs = [i for i in range(3) if true_o[i] != pred_o[i]]
        return len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1
    order_wrong = [r for r in type_correct if not r["order_correct"]]
    adj = sum(1 for r in order_wrong if is_adjacent_swap(r["true_order"], r["pred_order"]))
    stats["order_wrong_n"] = len(order_wrong)
    stats["order_wrong_adjacent_swap_fraction"] = adj / len(order_wrong) if order_wrong else None
    w(f"- among {len(order_wrong)} wrong-order examples: {adj} ({100*adj/len(order_wrong):.1f}%) are a single adjacent swap" if order_wrong else "- n/a")
    w()

    w("### 3e. Order accuracy by exact count of mild components present (0-3)")
    w("(tests whether the order-default-bias is independent of severity, as found in earlier")
    w("versions -- order barely improves even with zero mild components present)")
    for thresh, label in [(0, "sev<=0"), (1, "sev<=1")]:
        by_mild = collections.defaultdict(list)
        for r in type_correct:
            k = sum(1 for t in ["jpeg", "noise", "gamma"] if sev_idx(t, r["steps"][t]["true_value"]) <= thresh)
            by_mild[k].append(r["order_correct"])
        w(f"- {label}:")
        for k in range(4):
            if k in by_mild:
                w(f"    {k} mild component(s): order_accuracy={rate(by_mild[k]):.3f} (n={len(by_mild[k])})")
        stats.setdefault("order_by_mild_count", {})[label] = {str(k): {"accuracy": rate(v), "n": len(v)} for k, v in by_mild.items()}
    w()

    # ---- 4. recall/value_accuracy by own severity ----
    w("## 4. Recall and value_accuracy by own severity level (0=mildest)")
    for t in ["jpeg", "noise", "gamma"]:
        by_sev_recall = collections.defaultdict(list)
        by_sev_value = collections.defaultdict(list)
        for r in n3:
            idx = sev_idx(t, r["steps"][t]["true_value"])
            by_sev_recall[idx].append(t in r["pred_types"])
            by_sev_value[idx].append(r["steps"][t]["value_correct"])
        w(f"- {t}:")
        max_idx = 5 if t == "gamma" else 10
        for i in range(max_idx):
            if i in by_sev_recall:
                w(f"    idx={i}: recall={rate(by_sev_recall[i]):.3f}, value_acc={rate(by_sev_value[i]):.3f} (n={len(by_sev_recall[i])})")
    w()

    # ---- 5. cross-type interference: recall of A vs EACH other type's own severity, separately ----
    # reported per-neighbor rather than combined via max() -- combining two
    # types (one possibly gamma, capped at index 4) into one bucket via max()
    # ends up misattributing the effect to whichever type actually drove it
    w("## 5. Cross-type interference (recall of type A vs EACH other present type's own severity, separately)")
    w("(each of the two other types reported on its own native severity scale, independently --")
    w("NOT combined via max(), which would conflate the two types' contributions into one bucket")
    w("and let gamma's capped range silently hide behind a non-gamma neighbor's real severity)")
    interference = {}
    for target in ["jpeg", "noise", "gamma"]:
        others = [t for t in ["jpeg", "noise", "gamma"] if t != target]
        interference[target] = {}
        w(f"- recall of {target}, by each OTHER type's own severity level:")
        for other in others:
            by_sev = collections.defaultdict(list)
            for r in n3:
                oi = sev_idx(other, r["steps"][other]["true_value"])
                by_sev[oi].append(target in r["pred_types"])
            max_idx = 5 if other == "gamma" else 10
            interference[target][other] = {
                str(i): {"recall": rate(v), "n": len(v)} for i, v in sorted(by_sev.items())
            }
            w(f"    when co-occurring with {other}:")
            for i in range(max_idx):
                if i in by_sev:
                    w(f"      {other}_idx={i}: recall={rate(by_sev[i]):.3f} (n={len(by_sev[i])})")
    stats["cross_type_interference"] = interference
    w()

    # ---- 6. method accuracy ----
    w("## 6. Method accuracy per type")
    method_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        ma = rate([r["steps"][t]["method_correct"] for r in n3])
        mfa = rate([r["steps"][t]["method_family_correct"] for r in n3])
        method_stats[t] = {"n": len(n3), "method_accuracy": ma, "method_family_accuracy": mfa}
        w(f"- {t}: method_accuracy={ma:.3f}, method_family_accuracy={mfa:.3f} (n={len(n3)})")
    stats["method_accuracy"] = method_stats
    w()

    w("### 6b. Method accuracy GIVEN type-set AND order are both already correct")
    w("(the full-pipeline-correct endpoint: once the diagnosis and sequencing are right,")
    w("how good is the actual restoration method choice?)")
    both_correct = [r for r in n3 if r["type_set_correct"] and r["order_correct"]]
    w(f"- n with both type+order correct: {len(both_correct)}/{len(n3)} ({100*len(both_correct)/len(n3):.1f}%)")
    method_given_correct = {}
    for t in ["jpeg", "noise", "gamma"]:
        ma = rate([r["steps"][t]["method_correct"] for r in both_correct])
        mfa = rate([r["steps"][t]["method_family_correct"] for r in both_correct])
        method_given_correct[t] = {"n": len(both_correct), "method_accuracy": ma, "method_family_accuracy": mfa}
        w(f"- {t}: method_accuracy={ma:.3f}, method_family_accuracy={mfa:.3f}")
    stats["method_accuracy_given_type_and_order_correct"] = method_given_correct
    w()

    # ---- 7. value accuracy (unified) ----
    w("## 7. Value accuracy (unified severity-index definition)")
    value_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        exact = rate([r["steps"][t]["value_correct"] for r in n3])
        within1 = rate([r["steps"][t]["value_correct_within1"] for r in n3])
        value_stats[t] = {"n": len(n3), "exact_fraction": exact, "within1_fraction": within1}
        w(f"- {t}: exact={exact:.3f}, within-1-level={within1:.3f} (n={len(n3)})")
    stats["value_accuracy"] = value_stats
    w()

    # ---- 7b. within-1/within-2 value accuracy: strict miss vs graduated miss ----
    w("## 7b. Within-1 / within-2 value accuracy: strict miss vs graduated miss")
    w("(strict: a missed detection is always counted wrong, regardless of true severity --")
    w("same numbers as section 7. graduated: a missed detection is treated as an implicit")
    w("\"index -1\" guess, one step milder than the mildest real level.)")
    graduated_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        n = len(n3)
        exact = rate([r["steps"][t]["value_correct"] for r in n3])
        w1_strict = w2_strict = w1_grad = w2_grad = 0
        for r in n3:
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

    # ---- 7c. value accuracy excluding the weakest severity level(s) ----
    w("## 7c. Value accuracy excluding the weakest severity level(s)")
    w("(drops examples at/below the threshold entirely from both numerator and denominator --")
    w("a missed detection is NOT reinterpreted as a mild-level guess, it is simply excluded")
    w("here if its true severity is at/below the threshold)")
    excl_value_stats = {}
    for t in ["jpeg", "noise", "gamma"]:
        excl_value_stats[t] = {}
        w(f"- {t}:")
        exact_all = rate([r["steps"][t]["value_correct"] for r in n3])
        within1_all = rate([r["steps"][t]["value_correct_within1"] for r in n3])
        w(f"    all severities: exact={exact_all:.3f}, within-1-level={within1_all:.3f} (n={len(n3)})")
        excl_value_stats[t]["all"] = {"n": len(n3), "exact_fraction": exact_all, "within1_fraction": within1_all}
        for thresh in [0, 1]:
            sub = [r for r in n3 if sev_idx(t, r["steps"][t]["true_value"]) > thresh]
            exact = rate([r["steps"][t]["value_correct"] for r in sub])
            within1 = rate([r["steps"][t]["value_correct_within1"] for r in sub])
            w(f"    excl sev<={thresh}: exact={exact:.3f}, within-1-level={within1:.3f} (n={len(sub)})")
            excl_value_stats[t][f"excl_sev<={thresh}"] = {"n": len(sub), "exact_fraction": exact, "within1_fraction": within1}
    stats["value_accuracy_excluding_weak_severity"] = excl_value_stats
    w()

    # ---- 8. lenient type_set_accuracy (forgive ONE mild dropped type) ----
    w("## 8. Lenient type_set_accuracy (forgive ONE dropped type if its TRUE severity is mild)")

    def lenient_correct(r, mild_thresh):
        # forgives one dropped type if its true severity was mild -- no
        # "extra" direction here since all 3 types are always true
        true_t, pred_t = set(r["true_types"]), set(r["pred_types"])
        if true_t == pred_t:
            return True
        missing = true_t - pred_t
        extra = pred_t - true_t
        if len(missing) == 1 and len(extra) == 0:
            t = next(iter(missing))
            return sev_idx(t, r["steps"][t]["true_value"]) <= mild_thresh
        return False

    strict_acc = tsa
    lenient_stats = {}
    for thresh, label in [(0, "sev<=0"), (1, "sev<=1")]:
        la = rate([lenient_correct(r, thresh) for r in n3])
        lenient_stats[label] = la
        w(f"- strict={strict_acc:.3f} -> lenient ({label})={la:.3f}")
    stats["lenient_type_set_accuracy"] = lenient_stats
    w()

    # ---- 9. method memorization ----
    w("## 9. Method memorization check")
    memorization = {}
    for t in ["jpeg", "noise"]:
        true_c = collections.Counter(r["steps"][t]["true_method"] for r in n3)
        pred_c = collections.Counter(r["steps"][t]["pred_method"] for r in n3)
        n = len(n3)
        methods = sorted(set(true_c) | set(pred_c), key=lambda m: -pred_c.get(m, 0))
        memorization[t] = {m: {"true_pct": 100*true_c.get(m,0)/n, "pred_pct": 100*pred_c.get(m,0)/n} for m in methods}
        w(f"- {t}:")
        for m in methods:
            tp, pp = memorization[t][m]["true_pct"], memorization[t][m]["pred_pct"]
            flag = " <- over-used" if pp - tp > 5 else ""
            w(f"    {m}: true={tp:.1f}%  pred={pp:.1f}%{flag}")
    stats["method_memorization"] = memorization
    w()

    # ---- 10. excluding mild severity levels ----
    w("## 10. Effect of excluding the weakest severity level(s)")
    w("(excluded if ANY of the three true types is at or below the threshold)")
    for thresh, label in [(None, "all severities"), (0, "excl sev<=0"), (1, "excl sev<=1")]:
        if thresh is None:
            sub = n3
        else:
            sub = [r for r in n3 if all(sev_idx(t, r["steps"][t]["true_value"]) > thresh for t in ["jpeg", "noise", "gamma"])]
        sub_tsa = rate([r["type_set_correct"] for r in sub])
        tc = [r for r in sub if r["type_set_correct"]]
        oga = rate([r["order_correct"] for r in tc]) if tc else None
        w(f"- {label}: n={len(sub)}, type_set_accuracy={sub_tsa:.3f}" + (f", order_given_type={oga:.3f}" if oga is not None else ""))
    w()

    # ---- 11. SSIM impact ----
    w("## 11. SSIM impact of prediction errors (restricted to type-set-correct examples)")
    w("Ceiling = our own dataset's ground-truth pipeline (reverse of true corruption order,")
    w("reused single-distortion-optimal method) -- NOT a proven true optimum for combos.")
    if SKIP_SSIM:
        w("(SKIPPED -- SKIP_SSIM=1)")
        stats["ssim_impact"] = "skipped"
    else:
        deltas = []
        n_skipped = 0
        for i, r in enumerate(type_correct):
            base = r["id"].split("_final_")[0]
            img_rel = base_to_image.get(base)
            if img_rel is None:
                n_skipped += 1
                continue
            types = ["jpeg", "noise", "gamma"]
            true_values = {t: r["steps"][t]["true_value"] for t in types}
            true_methods = {t: r["steps"][t]["true_method"] for t in types}
            pred_methods = {t: r["steps"][t]["pred_method"] for t in types}
            pred_values = {t: r["steps"][t]["pred_value"] for t in types}
            if any(pred_methods[t] not in METHOD_FN[t] or pred_values[t] is None for t in types):
                n_skipped += 1
                continue
            img = cv2.imread(os.path.join(RAW_DIR, img_rel))
            if img is None:
                n_skipped += 1
                continue
            corruption_order = list(reversed(r["true_order"]))
            corrupted = corrupt_sequence(img, corruption_order, true_values)
            gt_restored = apply_pipeline(corrupted, r["true_order"], true_values, true_methods)
            ceiling_ssim = _ssim(img, gt_restored)
            try:
                pred_restored = apply_pipeline(corrupted, r["pred_order"], pred_values, pred_methods)
                achieved_ssim = _ssim(img, pred_restored)
            except Exception:
                n_skipped += 1
                continue
            deltas.append(ceiling_ssim - achieved_ssim)
            if (i + 1) % 100 == 0:
                print(f"  ssim progress: {i+1}/{len(type_correct)}", flush=True)
        stats["ssim_impact"] = {
            "n_compared": len(deltas), "n_skipped": n_skipped,
            "mean_ssim_loss": statistics.mean(deltas) if deltas else None,
            "median_ssim_loss": statistics.median(deltas) if deltas else None,
            "fraction_negligible_loss_lt_0.01": sum(1 for d in deltas if d < 0.01)/len(deltas) if deltas else None,
        }
        w(f"- compared={len(deltas)}, skipped={n_skipped}")
        if deltas:
            w(f"    mean SSIM lost vs ceiling: {statistics.mean(deltas):.4f}")
            w(f"    median SSIM lost vs ceiling: {statistics.median(deltas):.4f}")
            w(f"    fraction with negligible loss (<0.01): {stats['ssim_impact']['fraction_negligible_loss_lt_0.01']:.3f}")
    w()

    # ---- 12. does accuracy depend on the question phrasing? ----
    w("## 12. Does accuracy depend on the question phrasing?")
    w("(~20 phrasings used at random, independent of the true distortions/order.")
    w("NOTE: n per phrasing is small (~50) -- treat modest spread as noise.)")
    id_to_question = {e["id"]: e["conversations"][0]["value"].replace("<image>", "").strip() for e in test_manifest}
    by_question_type = collections.defaultdict(list)
    by_question_order = collections.defaultdict(list)
    for r in n3:
        q = id_to_question.get(r["id"], "(unknown)")
        by_question_type[q].append(r["type_set_correct"])
        if r["type_set_correct"]:
            by_question_order[q].append(r["order_correct"])
    q_stats = {}
    for q in by_question_type:
        q_stats[q] = {
            "n": len(by_question_type[q]),
            "type_set_accuracy": rate(by_question_type[q]),
            "order_accuracy_given_type_correct": rate(by_question_order.get(q, [])),
        }
    tsa_vals = [s["type_set_accuracy"] for s in q_stats.values()]
    w(f"- {len(q_stats)} distinct phrasings seen, ~{len(n3)//len(q_stats)} examples each on average")
    if len(tsa_vals) > 1:
        w(f"- type_set_accuracy across phrasings: min={min(tsa_vals):.3f}, max={max(tsa_vals):.3f}, "
          f"spread={max(tsa_vals)-min(tsa_vals):.3f}, stdev={statistics.stdev(tsa_vals):.3f}")
    w("- per-phrasing breakdown, worst to best (by type_set_accuracy):")
    for q, s in sorted(q_stats.items(), key=lambda kv: kv[1]["type_set_accuracy"]):
        short_q = q if len(q) <= 60 else q[:57] + "..."
        oga = s["order_accuracy_given_type_correct"]
        oga_str = f"{oga:.3f}" if oga is not None else "n/a"
        w(f"    type_set_acc={s['type_set_accuracy']:.3f}  order_given_type={oga_str}  (n={s['n']}): \"{short_q}\"")
    stats["accuracy_by_question_phrasing"] = q_stats
    w()

    # ---- 13. flagged examples ----
    flagged = []
    for r in wrong:
        base = r["id"].split("_final_")[0]
        img_rel = base_to_image.get(base)
        flagged.append({
            "id": r["id"], "base_id": base,
            "image_path": os.path.join(RAW_DIR, img_rel) if img_rel else None,
            "image_path_relative": img_rel,
            "pred_types": r["pred_types"],
            "true_order": r["true_order"], "pred_order": r["pred_order"],
            "output": r["output"],
        })
    stats["n_flagged_examples"] = len(flagged)
    w("## 13. Flagged examples")
    w(f"- {len(flagged)} wrong examples saved -> triples_flagged_examples.json")

    with open(os.path.join(OUT_DIR, "triples_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "triples_report.md"), "w") as f:
        f.write("\n".join(report))
    with open(os.path.join(OUT_DIR, "triples_flagged_examples.json"), "w") as f:
        json.dump(flagged, f, indent=2)

    print("\n".join(report))
    print(f"\nSaved -> {OUT_DIR}/triples_stats.json")
    print(f"Saved -> {OUT_DIR}/triples_report.md")
    print(f"Saved -> {OUT_DIR}/triples_flagged_examples.json")


if __name__ == "__main__":
    main()
