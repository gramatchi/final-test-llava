#!/bin/bash
#SBATCH -J analyze_pairs_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/analyze_pairs_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/analyze_pairs_final_err_%j.txt
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# pairs (n=2) category stats, ckpt28800, on the 1000 new photos
# (includes its own SSIM-impact section; set SKIP_SSIM=1 for a fast pass).

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/analyze_pairs_final.py
