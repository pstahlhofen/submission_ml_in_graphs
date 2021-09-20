#!/bin/bash
#SBATCH --nice
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=20
#SBATCH --array=1-1000%5

parout=($(echo ${!SLURM_ARRAY_TASK_ID} | sed 's/\([^[:space:]]*\)[[:space:]]*\([^[:space:]]*\).*/\1\n\2/'))
par=${parout[0]}
out=${parout[1]}
./eval_methods.py < $par > $out
