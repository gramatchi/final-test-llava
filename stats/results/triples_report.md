# v6 statistics: n=3 (all three distortions present: jpeg+noise+gamma)

## 1. Overview
- total examples: 1000
- unique photos: 1000

## 2. Type-set accuracy and confusion (only one true set: jpeg+noise+gamma)
- type_set_accuracy: 0.596
  predicted gamma+jpeg+noise: 596 (59.6%) [CORRECT]
  predicted gamma+jpeg: 110 (11.0%) [wrong]
  predicted gamma+noise: 81 (8.1%) [wrong]
  predicted jpeg+noise: 80 (8.0%) [wrong]
  predicted noise: 65 (6.5%) [wrong]
  predicted jpeg: 42 (4.2%) [wrong]
  predicted gamma: 21 (2.1%) [wrong]
  predicted (nothing): 5 (0.5%) [wrong]

### 2b. How many of the 3 true types get dropped, when wrong
  - 1 type(s) dropped: 271 (67.1% of wrong)
  - 2 type(s) dropped: 128 (31.7% of wrong)
  - 3 type(s) dropped: 5 (1.2% of wrong)
  - which type(s) get dropped (counting each dropped type once):
      gamma: 192
      noise: 178
      jpeg: 172

## 3. Order accuracy, bias, and per-position accuracy
- order_accuracy_given_type_correct: 0.419 (n=596)
### 3a. Accuracy by true order permutation, and most common wrong prediction
  gamma->jpeg->noise: n=101, acc=0.545, most common wrong -> jpeg->gamma->noise
  gamma->noise->jpeg: n=81, acc=0.519, most common wrong -> noise->gamma->jpeg
  jpeg->gamma->noise: n=92, acc=0.435, most common wrong -> gamma->jpeg->noise
  noise->gamma->jpeg: n=108, acc=0.370, most common wrong -> gamma->noise->jpeg
  noise->jpeg->gamma: n=119, acc=0.361, most common wrong -> noise->gamma->jpeg
  jpeg->noise->gamma: n=95, acc=0.316, most common wrong -> jpeg->gamma->noise

### 3b. Predicted-order bias (true frequency should be ~16.7% each)
  gamma->noise->jpeg: true=13.6%  pred=21.6%
  noise->gamma->jpeg: true=18.1%  pred=19.1%
  gamma->jpeg->noise: true=16.9%  pred=17.1%
  jpeg->gamma->noise: true=15.4%  pred=16.6%
  noise->jpeg->gamma: true=20.0%  pred=13.9%
  jpeg->noise->gamma: true=15.9%  pred=11.6%

### 3c. Position-wise accuracy (is each type correctly placed in position 1/2/3)
  position 1, type=gamma: n=182, accuracy=0.632
  position 1, type=jpeg: n=187, accuracy=0.684
  position 1, type=noise: n=227, accuracy=0.617
  position 2, type=gamma: n=200, accuracy=0.420
  position 2, type=jpeg: n=220, accuracy=0.491
  position 2, type=noise: n=176, accuracy=0.438
  position 3, type=gamma: n=214, accuracy=0.383
  position 3, type=jpeg: n=189, accuracy=0.757
  position 3, type=noise: n=193, accuracy=0.663

### 3d. Error shape: single adjacent swap vs fuller scramble
- among 346 wrong-order examples: 236 (68.2%) are a single adjacent swap

### 3e. Order accuracy by exact count of mild components present (0-3)
(tests whether the order-default-bias is independent of severity, as found in earlier
versions -- order barely improves even with zero mild components present)
- sev<=0:
    0 mild component(s): order_accuracy=0.445 (n=492)
    1 mild component(s): order_accuracy=0.296 (n=98)
    2 mild component(s): order_accuracy=0.333 (n=6)
- sev<=1:
    0 mild component(s): order_accuracy=0.463 (n=311)
    1 mild component(s): order_accuracy=0.390 (n=246)
    2 mild component(s): order_accuracy=0.237 (n=38)
    3 mild component(s): order_accuracy=1.000 (n=1)

## 4. Recall and value_accuracy by own severity level (0=mildest)
- jpeg:
    idx=0: recall=0.422, value_acc=0.012 (n=83)
    idx=1: recall=0.576, value_acc=0.182 (n=99)
    idx=2: recall=0.713, value_acc=0.287 (n=101)
    idx=3: recall=0.748, value_acc=0.215 (n=107)
    idx=4: recall=0.883, value_acc=0.213 (n=94)
    idx=5: recall=0.874, value_acc=0.432 (n=95)
    idx=6: recall=0.990, value_acc=0.635 (n=104)
    idx=7: recall=0.982, value_acc=0.523 (n=109)
    idx=8: recall=1.000, value_acc=0.770 (n=100)
    idx=9: recall=1.000, value_acc=0.861 (n=108)
- noise:
    idx=0: recall=0.176, value_acc=0.000 (n=102)
    idx=1: recall=0.573, value_acc=0.340 (n=103)
    idx=2: recall=0.693, value_acc=0.443 (n=88)
    idx=3: recall=0.872, value_acc=0.340 (n=94)
    idx=4: recall=0.915, value_acc=0.383 (n=94)
    idx=5: recall=1.000, value_acc=0.447 (n=103)
    idx=6: recall=0.983, value_acc=0.492 (n=118)
    idx=7: recall=0.989, value_acc=0.250 (n=88)
    idx=8: recall=1.000, value_acc=0.527 (n=110)
    idx=9: recall=1.000, value_acc=0.690 (n=100)
- gamma:
    idx=0: recall=0.425, value_acc=0.100 (n=200)
    idx=1: recall=0.742, value_acc=0.315 (n=213)
    idx=2: recall=0.898, value_acc=0.381 (n=176)
    idx=3: recall=0.986, value_acc=0.295 (n=217)
    idx=4: recall=0.995, value_acc=0.711 (n=194)

## 5. Cross-type interference (recall of type A vs EACH other present type's own severity, separately)
(each of the two other types reported on its own native severity scale, independently --
NOT combined via max(), which would conflate the two types' contributions into one bucket
and let gamma's capped range silently hide behind a non-gamma neighbor's real severity)
- recall of jpeg, by each OTHER type's own severity level:
    when co-occurring with noise:
      noise_idx=0: recall=0.863 (n=102)
      noise_idx=1: recall=0.893 (n=103)
      noise_idx=2: recall=0.898 (n=88)
      noise_idx=3: recall=0.862 (n=94)
      noise_idx=4: recall=0.851 (n=94)
      noise_idx=5: recall=0.816 (n=103)
      noise_idx=6: recall=0.763 (n=118)
      noise_idx=7: recall=0.807 (n=88)
      noise_idx=8: recall=0.791 (n=110)
      noise_idx=9: recall=0.760 (n=100)
    when co-occurring with gamma:
      gamma_idx=0: recall=0.710 (n=200)
      gamma_idx=1: recall=0.854 (n=213)
      gamma_idx=2: recall=0.830 (n=176)
      gamma_idx=3: recall=0.862 (n=217)
      gamma_idx=4: recall=0.881 (n=194)
- recall of noise, by each OTHER type's own severity level:
    when co-occurring with jpeg:
      jpeg_idx=0: recall=0.843 (n=83)
      jpeg_idx=1: recall=0.828 (n=99)
      jpeg_idx=2: recall=0.881 (n=101)
      jpeg_idx=3: recall=0.822 (n=107)
      jpeg_idx=4: recall=0.862 (n=94)
      jpeg_idx=5: recall=0.821 (n=95)
      jpeg_idx=6: recall=0.827 (n=104)
      jpeg_idx=7: recall=0.817 (n=109)
      jpeg_idx=8: recall=0.760 (n=100)
      jpeg_idx=9: recall=0.769 (n=108)
    when co-occurring with gamma:
      gamma_idx=0: recall=0.810 (n=200)
      gamma_idx=1: recall=0.836 (n=213)
      gamma_idx=2: recall=0.795 (n=176)
      gamma_idx=3: recall=0.820 (n=217)
      gamma_idx=4: recall=0.845 (n=194)
- recall of gamma, by each OTHER type's own severity level:
    when co-occurring with jpeg:
      jpeg_idx=0: recall=0.783 (n=83)
      jpeg_idx=1: recall=0.808 (n=99)
      jpeg_idx=2: recall=0.772 (n=101)
      jpeg_idx=3: recall=0.766 (n=107)
      jpeg_idx=4: recall=0.819 (n=94)
      jpeg_idx=5: recall=0.789 (n=95)
      jpeg_idx=6: recall=0.837 (n=104)
      jpeg_idx=7: recall=0.899 (n=109)
      jpeg_idx=8: recall=0.780 (n=100)
      jpeg_idx=9: recall=0.815 (n=108)
    when co-occurring with noise:
      noise_idx=0: recall=0.784 (n=102)
      noise_idx=1: recall=0.825 (n=103)
      noise_idx=2: recall=0.773 (n=88)
      noise_idx=3: recall=0.819 (n=94)
      noise_idx=4: recall=0.851 (n=94)
      noise_idx=5: recall=0.767 (n=103)
      noise_idx=6: recall=0.780 (n=118)
      noise_idx=7: recall=0.818 (n=88)
      noise_idx=8: recall=0.836 (n=110)
      noise_idx=9: recall=0.830 (n=100)

## 6. Method accuracy per type
- jpeg: method_accuracy=0.531, method_family_accuracy=0.729 (n=1000)
- noise: method_accuracy=0.522, method_family_accuracy=0.640 (n=1000)
- gamma: method_accuracy=0.808, method_family_accuracy=0.808 (n=1000)

### 6b. Method accuracy GIVEN type-set AND order are both already correct
(the full-pipeline-correct endpoint: once the diagnosis and sequencing are right,
how good is the actual restoration method choice?)
- n with both type+order correct: 250/1000 (25.0%)
- jpeg: method_accuracy=0.684, method_family_accuracy=0.916
- noise: method_accuracy=0.604, method_family_accuracy=0.764
- gamma: method_accuracy=1.000, method_family_accuracy=1.000

## 7. Value accuracy (unified severity-index definition)
- jpeg: exact=0.425, within-1-level=0.686 (n=1000)
- noise: exact=0.395, within-1-level=0.708 (n=1000)
- gamma: exact=0.356, within-1-level=0.707 (n=1000)

## 7b. Within-1 / within-2 value accuracy: strict miss vs graduated miss
(strict: a missed detection is always counted wrong, regardless of true severity --
same numbers as section 7. graduated: a missed detection is treated as an implicit
"index -1" guess, one step milder than the mildest real level.)
- jpeg: exact=0.425  within1(strict/graduated)=0.686/0.734  within2(strict/graduated)=0.772/0.862  (n=1000)
- noise: exact=0.395  within1(strict/graduated)=0.708/0.792  within2(strict/graduated)=0.784/0.912  (n=1000)
- gamma: exact=0.356  within1(strict/graduated)=0.707/0.822  within2(strict/graduated)=0.789/0.959  (n=1000)

## 7c. Value accuracy excluding the weakest severity level(s)
(drops examples at/below the threshold entirely from both numerator and denominator --
a missed detection is NOT reinterpreted as a mild-level guess, it is simply excluded
here if its true severity is at/below the threshold)
- jpeg:
    all severities: exact=0.425, within-1-level=0.686 (n=1000)
    excl sev<=0: exact=0.462, within-1-level=0.734 (n=917)
    excl sev<=1: exact=0.496, within-1-level=0.773 (n=818)
- noise:
    all severities: exact=0.395, within-1-level=0.708 (n=1000)
    excl sev<=0: exact=0.440, within-1-level=0.783 (n=898)
    excl sev<=1: exact=0.453, within-1-level=0.820 (n=795)
- gamma:
    all severities: exact=0.356, within-1-level=0.707 (n=1000)
    excl sev<=0: exact=0.420, within-1-level=0.809 (n=800)
    excl sev<=1: exact=0.458, within-1-level=0.857 (n=587)

## 8. Lenient type_set_accuracy (forgive ONE dropped type if its TRUE severity is mild)
- strict=0.596 -> lenient (sev<=0)=0.719
- strict=0.596 -> lenient (sev<=1)=0.794

## 9. Method memorization check
- jpeg:
    bilateral_d3: true=47.9%  pred=49.9%
    None: true=0.0%  pred=17.2% <- over-used
    bilateral_d5: true=20.5%  pred=16.7%
    bilateral_d9: true=20.1%  pred=13.4%
    nlm_tw5_sw11: true=8.3%  pred=2.7%
    bndry_k3: true=2.5%  pred=0.1%
    gauss_0.2x: true=0.7%  pred=0.0%
- noise:
    bilateral_d3: true=51.9%  pred=54.1%
    None: true=0.0%  pred=17.8% <- over-used
    bilateral_d21: true=7.1%  pred=8.2%
    nlm_tw5_sw11: true=10.3%  pred=5.9%
    median_k3: true=7.2%  pred=3.8%
    bilateral_d5: true=6.4%  pred=3.6%
    bilateral_d9: true=10.9%  pred=3.5%
    median_k15: true=1.9%  pred=2.0%
    median_k5: true=0.6%  pred=0.5%
    median_k7: true=0.9%  pred=0.5%
    bilateral_d15: true=2.1%  pred=0.1%
    median_k9: true=0.4%  pred=0.0%
    median_k11: true=0.3%  pred=0.0%

## 10. Effect of excluding the weakest severity level(s)
(excluded if ANY of the three true types is at or below the threshold)
- all severities: n=1000, type_set_accuracy=0.596, order_given_type=0.419
- excl sev<=0: n=660, type_set_accuracy=0.745, order_given_type=0.445
- excl sev<=1: n=369, type_set_accuracy=0.843, order_given_type=0.463

## 11. SSIM impact of prediction errors (restricted to type-set-correct examples)
Ceiling = our own dataset's ground-truth pipeline (reverse of true corruption order,
reused single-distortion-optimal method) -- NOT a proven true optimum for combos.
- compared=596, skipped=0
    mean SSIM lost vs ceiling: 0.0368
    median SSIM lost vs ceiling: 0.0189
    fraction with negligible loss (<0.01): 0.414

## 12. Does accuracy depend on the question phrasing?
(~20 phrasings used at random, independent of the true distortions/order.
NOTE: n per phrasing is small (~50) -- treat modest spread as noise.)
- 20 distinct phrasings seen, ~50 examples each on average
- type_set_accuracy across phrasings: min=0.435, max=0.809, spread=0.374, stdev=0.093
- per-phrasing breakdown, worst to best (by type_set_accuracy):
    type_set_acc=0.435  order_given_type=0.650  (n=46): "How would you clean this image up?"
    type_set_acc=0.468  order_given_type=0.409  (n=47): "Evaluate this image for combined distortions and recommen..."
    type_set_acc=0.490  order_given_type=0.280  (n=51): "Spot the issues here and tell me how to fix them, step by..."
    type_set_acc=0.500  order_given_type=0.591  (n=44): "What's wrong with this photo, and in what order should I ..."
    type_set_acc=0.509  order_given_type=0.429  (n=55): "Is this photo degraded? If so, how many ways, and what's ..."
    type_set_acc=0.531  order_given_type=0.462  (n=49): "Anything off with this pic? How do I fix it?"
    type_set_acc=0.540  order_given_type=0.333  (n=50): "Determine which distortions affect this image and the cor..."
    type_set_acc=0.542  order_given_type=0.281  (n=59): "What's wrong with this photo?"
    type_set_acc=0.590  order_given_type=0.583  (n=61): "This photo looks degraded in more than one way — what's g..."
    type_set_acc=0.603  order_given_type=0.229  (n=58): "Diagnose this picture — could be one issue or several — a..."
    type_set_acc=0.615  order_given_type=0.417  (n=39): "Tell me what's damaged in this photo and how to restore i..."
    type_set_acc=0.636  order_given_type=0.429  (n=44): "Assess this image for degradation and lay out the restora..."
    type_set_acc=0.640  order_given_type=0.531  (n=50): "Give me a diagnosis and a repair plan for this photo."
    type_set_acc=0.644  order_given_type=0.483  (n=45): "This image may have more than one problem. Diagnose it an..."
    type_set_acc=0.648  order_given_type=0.371  (n=54): "Inspect this image for artifacts. List every issue found ..."
    type_set_acc=0.655  order_given_type=0.395  (n=58): "Check this photo for problems."
    type_set_acc=0.676  order_given_type=0.348  (n=34): "Identify all distortions present and propose an ordered r..."
    type_set_acc=0.685  order_given_type=0.432  (n=54): "Analyze this image. Identify any distortions present and ..."
    type_set_acc=0.709  order_given_type=0.436  (n=55): "Walk me through fixing this image."
    type_set_acc=0.809  order_given_type=0.395  (n=47): "Fix this image."

## 13. Flagged examples
- 404 wrong examples saved -> triples_flagged_examples.json