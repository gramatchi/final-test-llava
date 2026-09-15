#!/bin/bash
#SBATCH -J example_grids_singles_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/example_grids_singles_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/example_grids_singles_final_err_%j.txt
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# 50 flagged (wrong) + 20 fully-correct example images for the
# singles (n=1) category, ckpt28800, on the 1000 new photos.
# NOTE: requires final_test/dataset/{jpeg,noise}_per_image/per_image.json
# (produced by run_find_best_methods_{jpeg,noise}_final.sh) for the "ideal
# method" SSIM ceiling caption -- run those first.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/make_example_grids_singles_final.py
