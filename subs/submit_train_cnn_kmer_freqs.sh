#!/bin/bash

set -u
export STDERR_DIR="./err"
export STDOUT_DIR="./out"
#init_dir "$STDERR_DIR" "$STDOUT_DIR"

#DIRS
BUG_DIR="/rsgrps/bhurwitz/taxonomic_class/data/BugMixingSim"
RESULTS_DIR="/rsgrps/bhurwitz/taxonomic_class/data/classification_results"


#PAIRS=("saprophyticus-pyogenes" "ecoli-shigella" "ecoli-saprophyticus")
PAIRS=("saprophyticus-pyogenes")
KMERS=(3)
EPOCHS=1000
READ="hiseq_100"


for i in ${PAIRS[@]}; do
    OUT_DIR="${RESULTS_DIR}/kmer_freqs/$READ/${i}"
    #init_dir "$OUT_DIR"
    for KMER in ${KMERS[@]}; do
        IN_FILE="${BUG_DIR}/kmer_freqs/${READ}/${KMER}/${i}-reads.txt"
        OUT_FILE="${OUT_DIR}/${KMER}mer_freqs_out.txt"
        FIG_FILE="${OUT_DIR}/${KMER}mer_freqs_out.png"
        export PY_ARGS="-k ${KMER} -ct -cv -i ${IN_FILE} -b 50 -e ${EPOCHS} -o ${OUT_FILE} -f ${FIG_FILE} -g"
        ARGS="-q standard -W group_list=bhurwitz -M mattmiller899@email.arizona.edu -m a"
        echo "$PY_ARGS"
        JOB_ID=`qsub $ARGS -v PY_ARGS -N kmer_freqs_${KMER}_${i} -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_kmer_freqs.sh`
        if [ "${JOB_ID}x" != "x" ]; then
            echo Job: \"$JOB_ID\"
        else
            echo Problem submitting job. Job terminated.
            exit 1
        fi
        echo "job successfully submitted"
    done
done
