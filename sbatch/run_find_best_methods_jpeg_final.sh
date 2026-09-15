#!/bin/bash
#SBATCH -J find_jpeg_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/find_jpeg_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/find_jpeg_final_err_%j.txt
#SBATCH -t 00:45:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G

# per-image best-restoration-method search (jpeg) for the 1000 new photos.
# ~15-16 min expected at 32 workers. Resumable.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /nfsd/lttm4/tesisti/gramatchi

N_WORKERS=32 python3 -u final_test/code/find_best_methods_jpeg_final.py
