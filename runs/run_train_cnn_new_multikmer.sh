#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --mem=168gb
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

set -u
#module load singularity/2/2.6.1
cd $SLURM_SUBMIT_DIR
echo "howdy"
echo "singularity exec --nv ~/singularity/pytorch.img python3.6 ./code/new_multikmer_classifier.py $PY_ARGS"
singularity exec --nv ~/singularity/pytorch.img python3.6 ./code/new_multikmer_classifier.py $PY_ARGS

