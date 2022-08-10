#PBS -l select=1:ncpus=28:mem=168gb
#PBS -l walltime=12:00:00


set -u
module load singularity/2/2.6.1
cd $PBS_O_WORKDIR/deep_learning

echo "singularity exec ~/extra/pytorch.img python3.6 ./classifier.py $PY_ARGS"
singularity exec ~/extra/pytorch.img python3.6 ./classifier.py $PY_ARGS
