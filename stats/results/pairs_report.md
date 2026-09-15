# v6 statistics: n=2 (exactly two distortions present)

## 1. Overview
- total examples: 1000
- unique photos: 1000

## 2. Type-set accuracy and confusion by pair-subset
- true=gamma+jpeg (n=366, acc=0.508):
    -> jpeg: 70 (19.1%)
    -> gamma: 55 (15.0%)
    -> gamma+jpeg+noise: 41 (11.2%)
- true=gamma+noise (n=341, acc=0.405):
    -> gamma+jpeg+noise: 77 (22.6%)
    -> noise: 66 (19.4%)
    -> gamma: 43 (12.6%)
- true=jpeg+noise (n=293, acc=0.352):
    -> noise: 60 (20.5%)
    -> gamma+jpeg+noise: 59 (20.1%)
    -> jpeg: 48 (16.4%)

## 3. Order accuracy and predicted-order bias
- order_accuracy_given_type_correct: 0.801 (n=427)
  gamma->jpeg: true=23.9%  pred=22.2%
  jpeg->gamma: true=19.7%  pred=21.3%
  gamma->noise: true=17.3%  pred=19.2%
  noise->gamma: true=15.0%  pred=13.1%
  jpeg->noise: true=13.1%  pred=12.6%
  noise->jpeg: true=11.0%  pred=11.5%

## 4. Error breakdown
- wrong: 573/1000 (57.3%)
  - dropped exactly one of the two true types: 370 (64.6%)
  - dropped both true types entirely: 26 (4.5%)
  - found both true types but ALSO hallucinated a 3rd: 177 (30.9%)

## 5. Over-detection (phantom 3rd type) rate by severity of the present distortions
(max severity of the two present types, both put on the same 0-9 scale -- gamma's 0-4
index is spread evenly across the range via gamma_to_10scale() rather than linearly
stretched, so it can land on any of the 10 buckets, not just {0,2,4,7,9})
  max_sev_bucket=0: over-detection rate=0.000 (n=13)
  max_sev_bucket=1: over-detection rate=0.029 (n=35)
  max_sev_bucket=2: over-detection rate=0.088 (n=57)
  max_sev_bucket=3: over-detection rate=0.115 (n=78)
  max_sev_bucket=4: over-detection rate=0.088 (n=91)
  max_sev_bucket=5: over-detection rate=0.164 (n=116)
  max_sev_bucket=6: over-detection rate=0.208 (n=120)
  max_sev_bucket=7: over-detection rate=0.148 (n=135)
  max_sev_bucket=8: over-detection rate=0.246 (n=183)
  max_sev_bucket=9: over-detection rate=0.262 (n=172)

## 6. Cross-type interference (recall of type A vs the OTHER present type's severity)
- recall of jpeg, by co-occurring type's severity:
    other type is mild: recall=0.860 (n=257)
    other type is mid: recall=0.770 (n=165)
    other type is strong: recall=0.722 (n=237)
- recall of noise, by co-occurring type's severity:
    other type is mild: recall=0.856 (n=271)
    other type is mid: recall=0.781 (n=151)
    other type is strong: recall=0.764 (n=212)
- recall of gamma, by co-occurring type's severity:
    other type is mild: recall=0.772 (n=298)
    other type is mid: recall=0.738 (n=206)
    other type is strong: recall=0.813 (n=203)

## 7. Recall and value_accuracy by own severity level (0=mildest)
- jpeg:
    idx=0: recall=0.231, value_acc=0.026 (n=78)
    idx=1: recall=0.507, value_acc=0.373 (n=75)
    idx=2: recall=0.824, value_acc=0.397 (n=68)
    idx=3: recall=0.726, value_acc=0.306 (n=62)
    idx=4: recall=0.906, value_acc=0.340 (n=53)
    idx=5: recall=0.942, value_acc=0.596 (n=52)
    idx=6: recall=0.929, value_acc=0.629 (n=70)
    idx=7: recall=0.984, value_acc=0.698 (n=63)
    idx=8: recall=1.000, value_acc=0.861 (n=72)
    idx=9: recall=1.000, value_acc=0.939 (n=66)
- noise:
    idx=0: recall=0.015, value_acc=0.000 (n=65)
    idx=1: recall=0.453, value_acc=0.266 (n=64)
    idx=2: recall=0.824, value_acc=0.544 (n=68)
    idx=3: recall=0.892, value_acc=0.462 (n=65)
    idx=4: recall=0.971, value_acc=0.414 (n=70)
    idx=5: recall=0.984, value_acc=0.444 (n=63)
    idx=6: recall=0.981, value_acc=0.415 (n=53)
    idx=7: recall=1.000, value_acc=0.197 (n=61)
    idx=8: recall=1.000, value_acc=0.612 (n=67)
    idx=9: recall=1.000, value_acc=0.690 (n=58)
- gamma:
    idx=0: recall=0.313, value_acc=0.075 (n=147)
    idx=1: recall=0.724, value_acc=0.313 (n=134)
    idx=2: recall=0.919, value_acc=0.453 (n=161)
    idx=3: recall=0.944, value_acc=0.286 (n=126)
    idx=4: recall=0.986, value_acc=0.655 (n=139)

## 8. Method accuracy per type
- jpeg: method_accuracy=0.484, method_family_accuracy=0.674 (n=659)
- noise: method_accuracy=0.491, method_family_accuracy=0.621 (n=634)
- gamma: method_accuracy=0.774, method_family_accuracy=0.774 (n=707)

### 8b. Method accuracy GIVEN type-set AND order are both already correct
(the full-pipeline-correct endpoint: once the diagnosis and sequencing are right,
how good is the actual restoration method choice?)
- n with both type+order correct: 342/1000 (34.2%)
- jpeg: method_accuracy=0.606, method_family_accuracy=0.843 (n=254)
- noise: method_accuracy=0.630, method_family_accuracy=0.790 (n=181)
- gamma: method_accuracy=1.000, method_family_accuracy=1.000 (n=249)

## 9. Value accuracy (unified severity-index definition -- exact index match, or within one adjacent level)
- jpeg: exact=0.511, within-1-level=0.727 (n=659)
- noise: exact=0.404, within-1-level=0.756 (n=634)
- gamma: exact=0.358, within-1-level=0.692 (n=707)

## 9b. Within-1 / within-2 value accuracy: strict miss vs graduated miss
(strict: a missed detection is always counted wrong, regardless of true severity --
same numbers as section 9. graduated: a missed detection is treated as an implicit
"index -1" guess, one step milder than the mildest real level.)
- jpeg: exact=0.511  within1(strict/graduated)=0.727/0.818  within2(strict/graduated)=0.766/0.914  (n=659)
- noise: exact=0.404  within1(strict/graduated)=0.756/0.856  within2(strict/graduated)=0.806/0.962  (n=634)
- gamma: exact=0.358  within1(strict/graduated)=0.692/0.835  within2(strict/graduated)=0.765/0.960  (n=707)

## 9c. Value accuracy excluding the weakest severity level(s)
(drops examples at/below the threshold entirely from both numerator and denominator --
a missed detection is NOT reinterpreted as a mild-level guess, it is simply excluded
here if its true severity is at/below the threshold)
- jpeg:
    all severities: exact=0.511, within-1-level=0.727 (n=659)
    excl sev<=0: exact=0.577, within-1-level=0.802 (n=581)
    excl sev<=1: exact=0.607, within-1-level=0.852 (n=506)
- noise:
    all severities: exact=0.404, within-1-level=0.756 (n=634)
    excl sev<=0: exact=0.450, within-1-level=0.840 (n=569)
    excl sev<=1: exact=0.473, within-1-level=0.891 (n=505)
- gamma:
    all severities: exact=0.358, within-1-level=0.692 (n=707)
    excl sev<=0: exact=0.432, within-1-level=0.809 (n=560)
    excl sev<=1: exact=0.469, within-1-level=0.854 (n=426)

## 10. Lenient type_set_accuracy (forgive ONE mild add/drop)
- strict=0.427 -> lenient (sev<=0)=0.634
- strict=0.427 -> lenient (sev<=1)=0.789

## 11. Method memorization check
- jpeg:
    bilateral_d3: true=46.9%  pred=43.1%
    None: true=0.0%  pred=21.2% <- over-used
    bilateral_d5: true=19.3%  pred=17.9%
    bilateral_d9: true=21.7%  pred=14.3%
    nlm_tw5_sw11: true=8.2%  pred=3.3%
    bndry_k3: true=3.0%  pred=0.2%
    median_k3: true=0.2%  pred=0.0%
    gauss_0.2x: true=0.8%  pred=0.0%
- noise:
    bilateral_d3: true=46.4%  pred=48.3%
    None: true=0.0%  pred=19.2% <- over-used
    bilateral_d21: true=7.9%  pred=6.9%
    nlm_tw5_sw11: true=13.9%  pred=6.8%
    bilateral_d5: true=6.2%  pred=5.0%
    bilateral_d9: true=12.0%  pred=4.9%
    median_k3: true=6.9%  pred=4.7%
    median_k15: true=2.5%  pred=3.0%
    median_k7: true=0.5%  pred=0.5%
    median_k5: true=1.3%  pred=0.5%
    bilateral_d15: true=2.2%  pred=0.2%
    median_k9: true=0.3%  pred=0.0%

## 12. Effect of excluding the weakest severity level(s)
(excluded if EITHER present type is at or below the threshold)
- all severities: n=1000, type_set_accuracy=0.427, order_given_type=0.801
- excl sev<=0: n=732, type_set_accuracy=0.531, order_given_type=0.807
- excl sev<=1: n=509, type_set_accuracy=0.582, order_given_type=0.851

## 13. SSIM impact of prediction errors (restricted to type-set-correct examples,
so predicted and true steps correspond 1:1 -- only order/method/value can differ)
Ceiling = our own dataset's ground-truth pipeline (reverse of true corruption order,
reused single-distortion-optimal method) -- NOT a proven true optimum for combos, see
project note: reused methods match the real combo-optimum only ~29% of the time.
- compared=427, skipped=0
    mean SSIM lost vs ceiling: 0.0306
    median SSIM lost vs ceiling: 0.0083
    fraction with negligible loss (<0.01): 0.508

## 14. Does accuracy depend on the question phrasing?
(~20 phrasings used at random, independent of the true distortions/order.
NOTE: n per phrasing is small (~50) -- treat modest spread as noise.)
- 20 distinct phrasings seen, ~50 examples each on average
- type_set_accuracy across phrasings: min=0.306, max=0.619, spread=0.313, stdev=0.072
- per-phrasing breakdown, worst to best (by type_set_accuracy):
    type_set_acc=0.306  order_given_type=0.727  (n=36): "Check this photo for problems."
    type_set_acc=0.333  order_given_type=0.727  (n=66): "How would you clean this image up?"
    type_set_acc=0.368  order_given_type=0.643  (n=38): "What's wrong with this photo?"
    type_set_acc=0.377  order_given_type=0.900  (n=53): "Tell me what's damaged in this photo and how to restore i..."
    type_set_acc=0.383  order_given_type=0.778  (n=47): "Give me a diagnosis and a repair plan for this photo."
    type_set_acc=0.400  order_given_type=0.727  (n=55): "Spot the issues here and tell me how to fix them, step by..."
    type_set_acc=0.409  order_given_type=0.815  (n=66): "Diagnose this picture — could be one issue or several — a..."
    type_set_acc=0.415  order_given_type=0.765  (n=41): "Is this photo degraded? If so, how many ways, and what's ..."
    type_set_acc=0.415  order_given_type=0.818  (n=53): "This image may have more than one problem. Diagnose it an..."
    type_set_acc=0.415  order_given_type=0.815  (n=65): "Fix this image."
    type_set_acc=0.417  order_given_type=0.750  (n=48): "Evaluate this image for combined distortions and recommen..."
    type_set_acc=0.426  order_given_type=1.000  (n=47): "Analyze this image. Identify any distortions present and ..."
    type_set_acc=0.429  order_given_type=0.917  (n=56): "This photo looks degraded in more than one way — what's g..."
    type_set_acc=0.432  order_given_type=0.895  (n=44): "Walk me through fixing this image."
    type_set_acc=0.435  order_given_type=0.700  (n=46): "Inspect this image for artifacts. List every issue found ..."
    type_set_acc=0.442  order_given_type=0.913  (n=52): "What's wrong with this photo, and in what order should I ..."
    type_set_acc=0.492  order_given_type=0.677  (n=63): "Identify all distortions present and propose an ordered r..."
    type_set_acc=0.512  order_given_type=0.857  (n=41): "Anything off with this pic? How do I fix it?"
    type_set_acc=0.561  order_given_type=0.609  (n=41): "Assess this image for degradation and lay out the restora..."
    type_set_acc=0.619  order_given_type=0.923  (n=42): "Determine which distortions affect this image and the cor..."

## 15. Flagged examples
- 573 wrong examples saved -> pairs_flagged_examples.json