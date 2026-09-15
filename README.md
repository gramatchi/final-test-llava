# Final test set — 1000 held-out photos

evaluation of the trained model (ckpt28800) on 1000 photos
that were never seen during training or any earlier eval run. 


## Test set design

No repeated photos within a category: each of the 1000 photos contributes
exactly one example to each of the 4 categories.

- 1000 examples with 0 distortions ("none")
- 1000 examples with 1 distortion ("singles")
- 1000 examples with 2 distortions ("pairs")
- 1000 examples with 3 distortions ("triples")

4000 total entries in `dataset/llava_final_test.json`.

## Pipeline (run in this order)

1. **Per-image best-restoration-method search** 
   `sbatch sbatch/run_find_best_methods_jpeg_final.sh` and
   `sbatch sbatch/run_find_best_methods_noise_final.sh`. Produces
   `dataset/{jpeg,noise}_per_image/per_image.json` (gamma needs no search —
   always restored via the `lut` method). Resumable.

2. **Build the method lookup + manifest** 
   ```
   cd /nfsd/lttm4/tesisti/gramatchi
   python3 final_test/code/build_method_lookup_final.py
   python3 final_test/code/build_dataset_manifest_final.py
   ```
   Produces `dataset/method_lookup_{jpeg,noise}.json` and
   `dataset/llava_final_test.json`.

3. **Run inference** (the expensive GPU step, ~2h15m-3h30m estimated for 4000
   examples): `sbatch sbatch/run_eval_final_test.sh`. Reuses the existing,
   unmodified `llava.eval.eval_multi_restoration_v6` engine. Produces `eval_results/final_test_pipeline_answers.jsonl` and
   `eval_results/final_test_pipeline_stats.json`. 

4. **Per-category statistics**:
   ```
   sbatch sbatch/run_analyze_none_final.sh
   sbatch sbatch/run_analyze_singles_final.sh
   sbatch sbatch/run_analyze_pairs_final.sh
   sbatch sbatch/run_analyze_triples_final.sh
   ```
   Each writes `stats/results/{category}_stats.json`, `{category}_report.md`,
   and `{category}_flagged_examples.json`: detection accuracy, method/value
   accuracy, lenient/graduated-miss variants, order accuracy, honest
   whole-pipeline SSIM, question-phrasing sensitivity, and more.

5. **Honest whole-pipeline SSIM for pairs/triples** (kept as separate scripts
   rather than folded into the analyze_* reports):
   ```
   sbatch sbatch/run_pairs_ssim_honest_final.sh
   sbatch sbatch/run_pairs_ssim_by_subset_final.sh
   sbatch sbatch/run_triples_ssim_honest_final.sh
   ```
   Writes `stats/results/pairs_ssim_honest.json`, `pairs_ssim_by_subset.json`,
   `triples_ssim_honest.json` (+ `triples_ssim_honest_deltas.jsonl`,
   incremental/resumable). Each also saves the 10 worst individual cases (by
   SSIM lost) to `stats/worst_cases/`.

6. **Example-image galleries** (run after the matching analyze_* script has
   produced `*_flagged_examples.json`, and after step 5 for the worst-case
   galleries):
   ```
   sbatch sbatch/run_make_example_grids_none_final.sh
   sbatch sbatch/run_make_example_grids_singles_final.sh   # needs step 1's per_image.json for the "ideal" SSIM caption
   sbatch sbatch/run_make_example_grids_pairs_final.sh
   sbatch sbatch/run_make_example_grids_triples_final.sh
   ```
   Writes 3-panel (original | corrupted | restored) PNGs to
   `example_images/{none,singles,pairs,triples}/`.

## Self-contained: this folder can be downloaded on its own

Every piece of code used to build/eval/analyze this test set lives under
`code/`, including its own copies of the small shared helper libraries
(`corruptors.py`, `methods_{jpeg,noise,gamma}.py`, `severity_levels.py`,
`severity_words.py` — plain numpy/cv2 functions, no other dependencies). They
import and run correctly on their own, with no other project files nearby.

The one exception is the actual model-inference engine
(`LLaVA/llava/eval/eval_multi_restoration_v6.py`, run by
`sbatch/run_eval_final_test.sh`) — this isn't portable, since it's part of
the full LLaVA package (needs `llava.model`/`llava.conversation`/
`llava.mm_utils`, PyTorch, transformers, the merged checkpoint, and a GPU). A
reference snapshot of it is kept at `code/model_inference_engine/` for
completeness (see the README there) — it can't be run from that location,
only from within the full LLaVA repo as `run_eval_final_test.sh` does.

## Folder layout

- `code/` — all orchestration scripts, plus the self-contained helper libraries
- `sbatch/` — SLURM job scripts wrapping each `code/*.py`
- `dataset/` — the generated manifest and per-image method lookups
- `eval_results/` — raw model answers + top-line stats
- `stats/results/` — per-category detailed stats, reports, flagged examples
- `stats/worst_cases/` — worst-by-SSIM-loss cases per category
- `example_images/` — 3-panel comparison galleries per category
