#!/bin/bash
#SBATCH -J pairs_ssim_subset_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/pairs_ssim_by_subset_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/pairs_ssim_by_subset_final_err_%j.txt
#SBATCH -t 00:40:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# same honest SSIM methodology as pairs_ssim_honest_final.py,
# broken down per pair-subset (jpeg+noise / jpeg+gamma / noise+gamma).

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/pairs_ssim_by_subset_final.py
