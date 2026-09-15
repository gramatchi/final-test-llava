#!/bin/bash
#SBATCH -J example_grids_triples_final
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/final_test/logs/example_grids_triples_final_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/final_test/logs/example_grids_triples_final_err_%j.txt
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p allgroups
#SBATCH --mem 8G

# 50 flagged (wrong) + up to 20 fully-correct example images for the
# triples category (the fully-correct pool tends to be small).

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi
python3 -u final_test/code/make_example_grids_triples_final.py
