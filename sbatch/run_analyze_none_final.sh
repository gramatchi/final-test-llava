#!/bin/bash
#SBATCH -J analyze_none_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/analyze_none_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/analyze_none_final_err_%j.txt
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# none (n=0) category stats, ckpt28800, on the 1000 new photos.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/analyze_none_final.py
