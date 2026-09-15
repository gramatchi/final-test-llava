#!/bin/bash
#SBATCH -J eval_final_test_rtx
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/eval_final_test_rtx_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/eval_final_test_rtx_err_%j.txt
#SBATCH -t 07:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:rtx:1

# Same job as run_eval_final_test.sh but requesting an RTX GPU instead of
# L40S -- submit both and whichever queue clears first wins. Writes to a
# separate answers/stats file (suffix _race) so the two jobs can't corrupt
# each other's output if both happen to start around the same time. Once one
# starts running, cancel the other while it's still pending.
#
# RTX 3090 has less VRAM than L40S but the model fits fine; expect it to run
# noticeably slower, hence the longer time limit.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export HF_HOME=/nfsd/lttm4/tesisti/gramatchi/.cache/huggingface
export PIP_CACHE_DIR=/nfsd/lttm4/tesisti/gramatchi/.cache/pip

cd /nfsd/lttm4/tesisti/gramatchi/LLaVA

DATA_DIR=/nfsd/lttm4/tesisti/gramatchi/final_test/dataset
RESULTS_DIR=/nfsd/lttm4/tesisti/gramatchi/final_test/eval_results
mkdir -p $RESULTS_DIR

echo "=== pipeline accuracy on the held-out final test set -- RTX ==="
python -m llava.eval.eval_multi_restoration_v6 \
    --model-path /nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6-final-28800 \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder /home/gramatchin/data/raw \
    --question-file $DATA_DIR/llava_final_test.json \
    --answers-file $RESULTS_DIR/final_test_pipeline_answers_race.jsonl \
    --stats-file $RESULTS_DIR/final_test_pipeline_stats_race.json

echo "=== done ==="
