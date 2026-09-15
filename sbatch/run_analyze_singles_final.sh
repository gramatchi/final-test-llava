#!/bin/bash
#SBATCH -J analyze_singles_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/analyze_singles_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/analyze_singles_final_err_%j.txt
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# singles (n=1) category stats -- includes the SSIM-impact section, so it
# gets a bit more time than none/pairs/triples.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/analyze_singles_final.py
