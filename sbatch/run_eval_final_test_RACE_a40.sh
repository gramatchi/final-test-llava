#!/bin/bash
#SBATCH -J eval_final_test_a40
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/eval_final_test_a40_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/eval_final_test_a40_err_%j.txt
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:a40:1

# Same idea as run_eval_final_test_RACE_rtx.sh, requesting an A40 instead --
# submit alongside the L40S/RTX versions and whichever queue clears first
# wins. Writes to its own answers/stats file (suffix _a40) so the jobs can't
# collide. Cancel the other pending ones once one actually starts.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export HF_HOME=/nfsd/lttm4/tesisti/gramatchi/.cache/huggingface
export PIP_CACHE_DIR=/nfsd/lttm4/tesisti/gramatchi/.cache/pip

cd /nfsd/lttm4/tesisti/gramatchi/LLaVA

DATA_DIR=/nfsd/lttm4/tesisti/gramatchi/final_test/dataset
RESULTS_DIR=/nfsd/lttm4/tesisti/gramatchi/final_test/eval_results
mkdir -p $RESULTS_DIR

echo "=== pipeline accuracy on the held-out final test set -- A40 ==="
python -m llava.eval.eval_multi_restoration_v6 \
    --model-path /nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6-final-28800 \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder /home/gramatchin/data/raw \
    --question-file $DATA_DIR/llava_final_test.json \
    --answers-file $RESULTS_DIR/final_test_pipeline_answers_a40.jsonl \
    --stats-file $RESULTS_DIR/final_test_pipeline_stats_a40.json

echo "=== done ==="
