#!/bin/bash

set -u
export STDERR_DIR="./err"
export STDOUT_DIR="./out"
#init_dir "$STDERR_DIR" "$STDOUT_DIR"

#DIRS
CHAR_CORPUS="/rsgrps/bhurwitz/taxonomic_class/data/char_corpus"
CENT_DIR="/rsgrps/bhurwitz/taxonomic_class/data/CentrifugeSim"
BUG_DIR="/rsgrps/bhurwitz/taxonomic_class/data/BugMixingSim"
RESULTS_DIR="/rsgrps/bhurwitz/taxonomic_class/data/classification_results"

#PARAMETERS
WIND=5
CONT_SIZE=100
SEQ_E=1000
POS_E=2000
SEQ_HD=5
POS_HD=5
EPOCHS=1000
BATCH=50
POS_FLAG=""
SEQ_FLAG=""
CONT_FLAG="-uc"
GPU_FLAG="-g"
SPLIT_FLAG=""



#TEST
#KMERS=(3)
#CONTIGS=("hiseq_100")
#ORGS=("ecoli-saprophyticus")
#TOKENS=("one_hot")


#LOOPERS
KMERS=(3 6)
CONTIGS=("hiseq_100")
ORGS=("saprophyticus-pyogenes" "ecoli-saprophyticus" "ecoli-shigella")
TOKENS=("one_hot")
ARGS="-q standard -W group_list=bhurwitz -M mattmiller899@email.arizona.edu -m a"
for READ in ${CONTIGS[@]}; do
    for ORG in ${ORGS[@]}; do
        OUT_DIR="${RESULTS_DIR}/${READ}/${ORG}"
        #init_dir "$OUT_DIR" "$OUT_DIR/loss_figs"
        for k in ${KMERS[@]}; do
            KMER=$k
            IN_FILE="${BUG_DIR}/kmers/${READ}/${KMER}/${ORG}-reads.txt"

            for l in ${TOKENS[@]}; do
                TOK_IN="${l}"
                TOK_EMB="${CHAR_CORPUS}/input/combined_${TOK_IN}_${KMER}mer_char_vals.txt"
                
                OUT_FILE="${OUT_DIR}/${KMER}mer_${TOK_IN}_${EPOCHS}eps.txt"
                FIG_FILE="${OUT_DIR}/loss_figs/${KMER}mer_${TOK_IN}_${EPOCHS}eps.png"
                export PY_ARGS="${GPU_FLAG} ${SPLIT_FLAG} -ct -cv -i ${IN_FILE} -c ${TOK_EMB} -b ${BATCH} -e ${EPOCHS} ${CONT_FLAG} -o ${OUT_FILE} -f ${FIG_FILE} -k ${KMER} -pa 2"
                #echo $PY_ARGS
                if [ "$GPU_FLAG" = "-g" ]; then
                    echo "Using GPU"
                    JOB_ID=`qsub $ARGS -v PY_ARGS -N ${KMER}_${l} -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_gpu.sh`
                else
                    echo "Using CPU"
                    JOB_ID=`qsub $ARGS -v PY_ARGS -N train_${KMER}_${l} -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_cpu.sh`
                fi
                if [ "${JOB_ID}x" != "x" ]; then
                    echo Job: \"$JOB_ID\"
                else
                    echo Problem submitting job. Job terminated.
                    exit 1
                fi
                echo "job successfully submitted"
            done
        done
    done
done

