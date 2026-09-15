# v6 statistics: n=0 ("none" -- no distortions present)

## 1. Overview
- total examples: 1000
- unique photos: 1000
- entries per photo: 1

## 2. Headline accuracy
- correctly says "no distortion": 0.8430 (843/1000)

## 3. Error breakdown by hallucination size
- wrong answers: 157/1000 (15.7%)
  - claims 1 distortion type(s): 143 (91.1% of wrong)
  - claims 2 distortion type(s): 12 (7.6% of wrong)
  - claims 3 distortion type(s): 2 (1.3% of wrong)

## 4. Which type gets hallucinated (single-type errors only)
- n=143
  - gamma: 122 (85.3%)
  - jpeg: 12 (8.4%)
  - noise: 9 (6.3%)

## 5. Claimed severity when hallucinating (0=mildest)
- jpeg (n=12): 0:1, 1:9, 2:0, 3:1, 4:0, 5:1, 6:0, 7:0, 8:0, 9:0
  -> 83.3% claimed at mild severity (index <= 1)
- noise (n=9): 0:0, 1:7, 2:2, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0
  -> 77.8% claimed at mild severity (index <= 1)
- gamma (n=122): 0:25, 1:77, 2:19, 3:1, 4:0
  -> 83.6% claimed at mild severity (index <= 1)

## 6. Brightness correlation check (does unusual overall brightness predict gamma hallucination?)
- population brightness (grayscale mean pixel value): mean=105.37, std=32.57
- photos with any gamma false-positive: 135
- mean |z-score| of brightness, flagged photos: 1.201
- mean |z-score| of brightness, ALL photos:     0.795
- flagged photos are 1.51x further from average brightness than a typical photo

## 7. Does accuracy depend on the question phrasing?
(~20 phrasings used at random, independent of the (here: always clean) ground truth.
NOTE: n per phrasing is small (~50) -- treat modest spread as noise.)
- 20 distinct phrasings seen, ~50 examples each on average
- accuracy across phrasings: min=0.760, max=0.912, spread=0.152, stdev=0.050
- per-phrasing breakdown, worst to best:
    0.760 (n=50): "Diagnose this picture — could be one issue or several — and tell me..."
    0.771 (n=48): "How would you clean this image up?"
    0.771 (n=48): "Check this photo for problems."
    0.784 (n=51): "This image may have more than one problem. Diagnose it and describe..."
    0.796 (n=49): "Analyze this image. Identify any distortions present and give a ste..."
    0.800 (n=55): "Give me a diagnosis and a repair plan for this photo."
    0.829 (n=41): "What's wrong with this photo, and in what order should I fix it?"
    0.830 (n=47): "Walk me through fixing this image."
    0.830 (n=53): "This photo looks degraded in more than one way — what's going on, a..."
    0.839 (n=56): "Spot the issues here and tell me how to fix them, step by step."
    0.845 (n=58): "Anything off with this pic? How do I fix it?"
    0.848 (n=33): "Tell me what's damaged in this photo and how to restore it properly."
    0.862 (n=58): "Determine which distortions affect this image and the correct order..."
    0.886 (n=44): "Identify all distortions present and propose an ordered restoration..."
    0.891 (n=55): "Fix this image."
    0.894 (n=47): "Inspect this image for artifacts. List every issue found and the re..."
    0.897 (n=58): "What's wrong with this photo?"
    0.902 (n=51): "Evaluate this image for combined distortions and recommend a full r..."
    0.902 (n=41): "Is this photo degraded? If so, how many ways, and what's the fix?"
    0.912 (n=57): "Assess this image for degradation and lay out the restoration steps..."

## 8. Lenient accuracy (forgive ONE hallucinated type if claimed at mild severity)
(strict = section 2's exact pred_types==[] requirement. lenient = also count it correct
if the model named exactly one extra type, as long as its OWN claimed severity for that
type is mild (index <= threshold). A 2-3 type hallucination is never forgiven here.)
- strict=0.843 -> lenient (sev<=0)=0.869
- strict=0.843 -> lenient (sev<=1)=0.962

## 9. SSIM impact of hallucinated "fixes" applied to an already-clean photo
(restricted to single-type hallucinations so exactly one restoration step is well-defined.
Ceiling is trivially 1.0 -- doing nothing to a clean photo is already perfect. "Achieved"
applies the model's OWN claimed method+value to the clean original -- what a deployed
system would actually hand the user if it trusted this false alarm.)
- n=143 (of 143 single-type hallucinations; 0 skipped -- image not found)
- mean SSIM lost by "fixing" a clean photo:   0.1804
- median SSIM lost:                            0.1543
- max SSIM lost (worst single case):           0.6831
- top 10 worst individual cases saved -> stats/worst_cases/none_ssim_worst_cases.json

## 10. Flagged examples
- 157 wrong examples saved with full detail (image path, output text, brightness z-score) -> see none_flagged_examples.json
