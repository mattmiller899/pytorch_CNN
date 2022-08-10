#PBS -l select=1:ncpus=28:mem=168gb:ngpus=1
#PBS -l walltime=24:00:00


set -u
module load singularity/2/2.6.1
cd $PBS_O_WORKDIR/deep_learning
echo "singularity exec --nv ~/extra/pytorch.img python3.6 ./kmer_freq_classifier.py $PY_ARGS"
singularity exec --nv ~/extra/pytorch.img python3.6 ./kmer_freq_classifier.py $PY_ARGS
