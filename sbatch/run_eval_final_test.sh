#!/bin/bash
#SBATCH -J eval_final_test
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/eval_final_test_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/eval_final_test_err_%j.txt
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:l40s:1

# runs the trained checkpoint on the 1000 held-out photos via the
# existing eval_multi_restoration_v6 engine. 4000 examples, expect
# roughly 2h15m-3h30m on an L40S.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export HF_HOME=/nfsd/lttm4/tesisti/gramatchi/.cache/huggingface
export PIP_CACHE_DIR=/nfsd/lttm4/tesisti/gramatchi/.cache/pip

cd /nfsd/lttm4/tesisti/gramatchi/LLaVA

DATA_DIR=/nfsd/lttm4/tesisti/gramatchi/final_test/dataset
RESULTS_DIR=/nfsd/lttm4/tesisti/gramatchi/final_test/eval_results
mkdir -p $RESULTS_DIR

echo "=== pipeline accuracy on the held-out final test set (1000 photos) ==="
python -m llava.eval.eval_multi_restoration_v6 \
    --model-path /nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6-final-28800 \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder /home/gramatchin/data/raw \
    --question-file $DATA_DIR/llava_final_test.json \
    --answers-file $RESULTS_DIR/final_test_pipeline_answers.jsonl \
    --stats-file $RESULTS_DIR/final_test_pipeline_stats.json

echo "=== done ==="
