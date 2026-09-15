#!/bin/bash
#SBATCH -J pairs_ssim_honest_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/pairs_ssim_honest_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/pairs_ssim_honest_final_err_%j.txt
#SBATCH -t 00:40:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# honest whole-pipeline SSIM-impact computation across all 1000
# pairs examples (ckpt28800), on the new photos.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/pairs_ssim_honest_final.py
