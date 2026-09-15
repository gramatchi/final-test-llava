# v6 statistics: n=1 (exactly one distortion present)

## 1. Overview
- total examples: 1000
- unique photos: 1000
- true type = jpeg: 324
- true type = noise: 358
- true type = gamma: 318

## 2. Recall per type (correctly includes the true type)
- jpeg: recall=0.873 (n=324)
- noise: recall=0.832 (n=358)
- gamma: recall=0.761 (n=318)

## 3. Error breakdown (when true type is NOT correctly/exactly matched)
- wrong: 326/1000 (32.6%)
  - true type missed entirely (not in pred at all): 177 (54.3% of wrong)
  - true type found, but extra type(s) also hallucinated: 149 (45.7% of wrong)

### 3b. Which extra type gets hallucinated (over-detection cases, exactly 1 extra)
  - extra=gamma: 78 (66.1%)
  - extra=jpeg: 31 (26.3%)
  - extra=noise: 9 (7.6%)

### 3c. Over-detection rate by the TRUE distortion's OWN severity level
(does a stronger real distortion make the model more likely to ALSO hallucinate an
extra, phantom type on top of it? tests the opposite end of the mild-severity story:
mild severity -> more MISSES of the real type; strong severity -> more EXTRA
hallucinated types, checked here.)
- jpeg:
    sev_idx=0: over-detection rate=0.000 (n=27)
    sev_idx=1: over-detection rate=0.081 (n=37)
    sev_idx=2: over-detection rate=0.036 (n=28)
    sev_idx=3: over-detection rate=0.115 (n=26)
    sev_idx=4: over-detection rate=0.103 (n=39)
    sev_idx=5: over-detection rate=0.120 (n=25)
    sev_idx=6: over-detection rate=0.100 (n=40)
    sev_idx=7: over-detection rate=0.242 (n=33)
    sev_idx=8: over-detection rate=0.293 (n=41)
    sev_idx=9: over-detection rate=0.071 (n=28)
- noise:
    sev_idx=0: over-detection rate=0.000 (n=38)
    sev_idx=1: over-detection rate=0.111 (n=36)
    sev_idx=2: over-detection rate=0.405 (n=37)
    sev_idx=3: over-detection rate=0.209 (n=43)
    sev_idx=4: over-detection rate=0.286 (n=28)
    sev_idx=5: over-detection rate=0.226 (n=31)
    sev_idx=6: over-detection rate=0.387 (n=31)
    sev_idx=7: over-detection rate=0.333 (n=33)
    sev_idx=8: over-detection rate=0.350 (n=40)
    sev_idx=9: over-detection rate=0.195 (n=41)
- gamma:
    sev_idx=0: over-detection rate=0.030 (n=67)
    sev_idx=1: over-detection rate=0.000 (n=56)
    sev_idx=2: over-detection rate=0.081 (n=62)
    sev_idx=3: over-detection rate=0.134 (n=67)
    sev_idx=4: over-detection rate=0.076 (n=66)

### 3d. Type confusion (true type missed AND a different type predicted in its place)
(splits the 'missed entirely' bucket from section 3 into two qualitatively different
failures: SILENT miss -- the model says the photo is clean -- vs TYPE CONFUSION -- the
model is confident something is wrong, it just names the wrong distortion family.)
- of 177 missed-entirely cases:
  - silent miss (predicted no distortion at all): 156 (88.1%)
  - type confusion (named a different type instead): 21 (11.9%)

  confusion matrix (true type -> what was predicted instead, type-confusion cases only):
    true=jpeg (8 confused examples):
      -> predicted gamma: 8 (100.0%)
    true=noise (9 confused examples):
      -> predicted gamma: 8 (88.9%)
      -> predicted jpeg: 1 (11.1%)
    true=gamma (4 confused examples):
      -> predicted jpeg: 2 (50.0%)
      -> predicted noise: 2 (50.0%)

### 3e. Silent-miss rate by severity level (model said "no distortion" entirely)
(narrower than 1-recall in section 4, which also counts type-confusion misses --
this counts ONLY pred_types == [] cases, i.e. the model saw nothing wrong at all)
- jpeg:
    idx=0: silent_miss_rate=0.741 (n=27)
    idx=1: silent_miss_rate=0.216 (n=37)
    idx=2: silent_miss_rate=0.071 (n=28)
    idx=3: silent_miss_rate=0.077 (n=26)
    idx=4: silent_miss_rate=0.026 (n=39)
    idx=5: silent_miss_rate=0.000 (n=25)
    idx=6: silent_miss_rate=0.000 (n=40)
    idx=7: silent_miss_rate=0.000 (n=33)
    idx=8: silent_miss_rate=0.000 (n=41)
    idx=9: silent_miss_rate=0.000 (n=28)
- noise:
    idx=0: silent_miss_rate=0.842 (n=38)
    idx=1: silent_miss_rate=0.500 (n=36)
    idx=2: silent_miss_rate=0.027 (n=37)
    idx=3: silent_miss_rate=0.000 (n=43)
    idx=4: silent_miss_rate=0.000 (n=28)
    idx=5: silent_miss_rate=0.000 (n=31)
    idx=6: silent_miss_rate=0.000 (n=31)
    idx=7: silent_miss_rate=0.000 (n=33)
    idx=8: silent_miss_rate=0.000 (n=40)
    idx=9: silent_miss_rate=0.000 (n=41)
- gamma:
    idx=0: silent_miss_rate=0.672 (n=67)
    idx=1: silent_miss_rate=0.375 (n=56)
    idx=2: silent_miss_rate=0.081 (n=62)
    idx=3: silent_miss_rate=0.015 (n=67)
    idx=4: silent_miss_rate=0.000 (n=66)

## 4. Recall and value_accuracy by severity level (0=mildest)
- jpeg:
    idx=0: recall=0.148, value_acc=0.037 (n=27)
    idx=1: recall=0.730, value_acc=0.703 (n=37)
    idx=2: recall=0.929, value_acc=0.607 (n=28)
    idx=3: recall=0.923, value_acc=0.577 (n=26)
    idx=4: recall=0.949, value_acc=0.744 (n=39)
    idx=5: recall=0.960, value_acc=0.840 (n=25)
    idx=6: recall=0.975, value_acc=0.850 (n=40)
    idx=7: recall=1.000, value_acc=0.909 (n=33)
    idx=8: recall=1.000, value_acc=0.951 (n=41)
    idx=9: recall=1.000, value_acc=0.964 (n=28)
- noise:
    idx=0: recall=0.000, value_acc=0.000 (n=38)
    idx=1: recall=0.444, value_acc=0.389 (n=36)
    idx=2: recall=0.973, value_acc=0.703 (n=37)
    idx=3: recall=1.000, value_acc=0.419 (n=43)
    idx=4: recall=0.964, value_acc=0.500 (n=28)
    idx=5: recall=1.000, value_acc=0.581 (n=31)
    idx=6: recall=1.000, value_acc=0.516 (n=31)
    idx=7: recall=1.000, value_acc=0.121 (n=33)
    idx=8: recall=1.000, value_acc=0.675 (n=40)
    idx=9: recall=1.000, value_acc=0.780 (n=41)
- gamma:
    idx=0: recall=0.284, value_acc=0.000 (n=67)
    idx=1: recall=0.625, value_acc=0.375 (n=56)
    idx=2: recall=0.919, value_acc=0.371 (n=62)
    idx=3: recall=0.970, value_acc=0.313 (n=67)
    idx=4: recall=1.000, value_acc=0.697 (n=66)

## 5. Method accuracy per type
- jpeg: method_accuracy=0.611, method_family_accuracy=0.790 (n=324)
- noise: method_accuracy=0.528, method_family_accuracy=0.654 (n=358)
- gamma: method_accuracy=0.761, method_family_accuracy=0.761 (n=318)

### 5b. Method accuracy GIVEN the type-set is cleanly correct (no drop, no extra hallucination)
(order is trivial for a single distortion, so this is the singles analog of the
pairs/triples 'given type+order correct' full-pipeline-correct endpoint)
- n with clean type_set_correct: 674/1000 (67.4%)
- jpeg: method_accuracy=0.691, method_family_accuracy=0.901 (n=243)
- noise: method_accuracy=0.638, method_family_accuracy=0.790 (n=210)
- gamma: method_accuracy=1.000, method_family_accuracy=1.000 (n=221)

## 6. Value accuracy (unified severity-index definition, project decision 2026-09-02:
replaces the old per-type rules where jpeg required an exact integer match while
noise/gamma had generous, never-justified tolerances -- now all three types are judged
on the same 10-level yardstick: exact index match, or within one adjacent level.)
- jpeg: exact=0.738, within-1-level=0.870 (n=324)
- noise: exact=0.472, within-1-level=0.782 (n=358)
- gamma: exact=0.349, within-1-level=0.689 (n=318)

## 6b. Value accuracy excluding the weakest severity level(s)
(drops examples at/below the threshold entirely from both numerator and
denominator -- a missed detection is NOT reinterpreted as a mild-level guess,
it is simply excluded here if its true severity is at/below the threshold)
- jpeg:
    all severities: exact=0.738, within-1-level=0.870 (n=324)
    excl sev<=0: exact=0.801, within-1-level=0.936 (n=297)
    excl sev<=1: exact=0.815, within-1-level=0.965 (n=260)
- noise:
    all severities: exact=0.472, within-1-level=0.782 (n=358)
    excl sev<=0: exact=0.528, within-1-level=0.875 (n=320)
    excl sev<=1: exact=0.546, within-1-level=0.930 (n=284)
- gamma:
    all severities: exact=0.349, within-1-level=0.689 (n=318)
    excl sev<=0: exact=0.442, within-1-level=0.821 (n=251)
    excl sev<=1: exact=0.462, within-1-level=0.892 (n=195)

## 6c. Within-1 / within-2 value accuracy: strict miss vs graduated miss
(strict: a missed detection is always counted wrong, regardless of true
severity -- same numbers as section 6/6b. graduated: a missed detection is
treated as an implicit "index -1" guess, one step milder than the mildest
real level, so missing a very mild true distortion counts as a small
distance-1 error instead of an unqualified failure.)
- jpeg: exact=0.738  within1(strict/graduated)=0.870/0.941  within2(strict/graduated)=0.873/0.975  (n=324)
- noise: exact=0.472  within1(strict/graduated)=0.782/0.888  within2(strict/graduated)=0.830/0.992  (n=358)
- gamma: exact=0.349  within1(strict/graduated)=0.689/0.840  within2(strict/graduated)=0.745/0.962  (n=318)

## 7. Lenient type_set_accuracy (forgive one mild add OR drop)
("extra-only" forgives ONLY the over-detection case -- true type found, one extra
hallucinated at mild severity -- same definition as before. "symmetric" ALSO forgives
the opposite failure: the true type dropped entirely (pred_types==[]), judged by ITS OWN
true severity rather than a claimed one (there's nothing claimed for a type never
mentioned) -- same convention pairs/triples already use for their dropped-type case.)
- strict=0.674 -> lenient (sev<=0): extra_only=0.693, symmetric=0.790
- strict=0.674 -> lenient (sev<=1): extra_only=0.748, symmetric=0.892

## 8. Method memorization check (true frequency vs predicted frequency)
- jpeg:
    bilateral_d3: true=48.8%  pred=50.0%
    bilateral_d5: true=21.0%  pred=19.1%
    bilateral_d9: true=19.4%  pred=15.1%
    None: true=0.0%  pred=12.7% <- over-used
    nlm_tw5_sw11: true=7.7%  pred=2.8%
    bndry_k3: true=2.2%  pred=0.3%
    gauss_0.2x: true=0.9%  pred=0.0%
- noise:
    bilateral_d3: true=51.4%  pred=56.1%
    None: true=0.0%  pred=16.8% <- over-used
    bilateral_d21: true=7.0%  pred=9.2%
    nlm_tw5_sw11: true=8.9%  pred=5.0%
    median_k3: true=6.4%  pred=3.4%
    bilateral_d5: true=8.1%  pred=3.4%
    bilateral_d9: true=10.9%  pred=3.1%
    median_k15: true=1.7%  pred=2.5%
    median_k7: true=0.6%  pred=0.3%
    median_k5: true=0.8%  pred=0.3%
    median_k9: true=0.3%  pred=0.0%
    bilateral_d15: true=3.9%  pred=0.0%

## 9. SSIM impact of prediction errors (jpeg/noise/gamma -- real method choice)
For each example: corrupt the real photo at the TRUE severity, restore using the
model's PREDICTED method+value (what a deployed system would actually do not knowing
ground truth), measure SSIM vs the clean original. Compare to the best achievable SSIM
(best method at the true severity). If the model gave no usable prediction (type missed,
or method/value unparseable), 'achieved' = the UNRESTORED corrupted image -- this case is
counted in the overall average, not dropped, since that's what a real deployed system
would actually leave the user with.
- jpeg: n=324 (of which 283 had a usable prediction, 41 got no restoration applied; 0 truly unsimulable)
    OVERALL (all 324): mean SSIM lost vs ideal: 0.0021, median: 0.0000, fraction negligible (<0.01): 0.966
    -- with prediction (283): mean lost 0.0019, median 0.0000
    -- no restoration applied (41): mean lost 0.0035, median 0.0023
- noise: n=358 (of which 298 had a usable prediction, 60 got no restoration applied; 0 truly unsimulable)
    OVERALL (all 358): mean SSIM lost vs ideal: 0.0102, median: 0.0001, fraction negligible (<0.01): 0.771
    -- with prediction (298): mean lost 0.0089, median 0.0001
    -- no restoration applied (60): mean lost 0.0166, median 0.0001
- gamma: n=318 (of which 242 had a usable prediction, 76 got no restoration applied; 0 truly unsimulable)
    OVERALL (all 318): mean SSIM lost vs ideal: 0.0683, median: 0.0403, fraction negligible (<0.01): 0.352
    -- with prediction (242): mean lost 0.0438, median 0.0229
    -- no restoration applied (76): mean lost 0.1464, median 0.0963
- top 10 worst individual cases (by SSIM lost, across all 3 types) saved -> stats/worst_cases/singles_ssim_worst_cases.json

## 10. Does accuracy depend on the question phrasing?
(~20 different phrasings of the same request are used, chosen at random per example --
independent of the actual distortion. If wording doesn't matter, type_set_accuracy should
be roughly flat across all of them. NOTE: with ~1000 examples over ~20 phrasings, n per
phrasing is small (~50) -- treat modest spread as noise, only a large spread as a real signal.)
- 20 distinct phrasings seen, ~50 examples each on average
- type_set_accuracy across phrasings: min=0.509, max=0.804, spread=0.294, stdev=0.071
- per-phrasing breakdown, worst to best:
    0.509 (n=53): "What's wrong with this photo, and in what order should I fix it?"
    0.587 (n=46): "Tell me what's damaged in this photo and how to restore it properly."
    0.614 (n=57): "Analyze this image. Identify any distortions present and give a ste..."
    0.615 (n=39): "Assess this image for degradation and lay out the restoration steps..."
    0.620 (n=50): "Evaluate this image for combined distortions and recommend a full r..."
    0.622 (n=45): "Identify all distortions present and propose an ordered restoration..."
    0.627 (n=51): "Fix this image."
    0.649 (n=57): "This photo looks degraded in more than one way — what's going on, a..."
    0.653 (n=49): "Check this photo for problems."
    0.660 (n=53): "Walk me through fixing this image."
    0.698 (n=63): "Inspect this image for artifacts. List every issue found and the re..."
    0.702 (n=47): "Anything off with this pic? How do I fix it?"
    0.706 (n=51): "How would you clean this image up?"
    0.714 (n=56): "This image may have more than one problem. Diagnose it and describe..."
    0.725 (n=40): "Diagnose this picture — could be one issue or several — and tell me..."
    0.732 (n=41): "Give me a diagnosis and a repair plan for this photo."
    0.735 (n=49): "Is this photo degraded? If so, how many ways, and what's the fix?"
    0.737 (n=57): "What's wrong with this photo?"
    0.775 (n=40): "Spot the issues here and tell me how to fix them, step by step."
    0.804 (n=56): "Determine which distortions affect this image and the correct order..."

## 11. Flagged examples
- 326 wrong examples saved -> singles_flagged_examples.json
