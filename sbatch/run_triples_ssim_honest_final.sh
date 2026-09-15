#!/bin/bash
#SBATCH -J triples_ssim_honest_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/triples_ssim_honest_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/triples_ssim_honest_final_err_%j.txt
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# honest whole-pipeline SSIM-impact computation across all 1000
# triples examples (ckpt28800), 3 sequential steps per example -- the most
# expensive of the SSIM-impact scripts. Saves incrementally (one line per
# example) to triples_ssim_honest_deltas.jsonl in final_test/stats/results/,
# so a rerun of this exact script resumes instead of restarting if the job
# times out or dies partway through.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/triples_ssim_honest_final.py
