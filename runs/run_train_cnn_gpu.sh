#PBS -l select=1:ncpus=28:mem=168gb:ngpus=1
#PBS -l walltime=24:00:00


set -u
#module load singularity/2/2.6.1
module load singularity
cd $PBS_O_WORKDIR/deep_learning

echo "singularity exec --nv ~/extra/pytorch.img python3.6 ./classifier.py $PY_ARGS"
singularity exec --nv ~/extra/pytorch.img python3.6 ./classifier.py $PY_ARGS

