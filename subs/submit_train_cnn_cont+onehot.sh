#!/bin/bash

set -u
export STDERR_DIR="./err"
export STDOUT_DIR="./out"
#init_dir "$STDERR_DIR" "$STDOUT_DIR"

#DIRS
CHAR_CORPUS="/rsgrps/bhurwitz/taxonomic_class/data/char_corpus"
CENT_DIR="/rsgrps/bhurwitz/taxonomic_class/data/CentrifugeSim"
BUG_DIR="/rsgrps/bhurwitz/taxonomic_class/data/BugMixingSim"
MULTIKMER_DIR="/rsgrps/bhurwitz/taxonomic_class/data/test_everything"
INPUT_DIR="${MULTIKMER_DIR}/in"
CONT_DIR="${MULTIKMER_DIR}/cont_in"

POS_DIR="${MULTIKMER_DIR}/pos_in_one_hot"
#POS_DIR="${MULTIKMER_DIR}/pos_in"
SEQ_DIR="${MULTIKMER_DIR}/seq_in"
#TODO CHANGE BACK
RESULTS_DIR="${MULTIKMER_DIR}/results"

#PARAMETERS
EPOCHS=100
BATCH=50
KFOLD=4

#LOOPERS
declare -a KMERS=("3")
CONTIGS=("hiseq_100")
ORGS=("saprophyticus-pyogenes")
#TOKENS=("one_hot")

#TODO CHANGE BACK
FLAG="-up -g -sc -uc"
FILTERS=(3)
FCS=(2)
#FILTERS=(3)
#FCS=(2)
CONVS=(2)
PATIENCES=(2)

ARGS="-q standard -W group_list=bhurwitz -M mattmiller899@email.arizona.edu -m a"
for READ in ${CONTIGS[@]}; do
    for ORG in ${ORGS[@]}; do
        OUT_DIR="${RESULTS_DIR}/${READ}/${ORG}"
        IN_DIR="${INPUT_DIR}/${READ}/${ORG}"
        #init_dir "$OUT_DIR"
        #IN_FILE="${BUG_DIR}/kmers/${READ}/${KMER}/${ORG}-reads.txt"
        for KMER in "${KMERS[@]}"; do #Surround in quotes so that spaces in the strings dont get split
            KMERSTR=${KMER//\ /-}
            for FILTER in ${FILTERS[@]}; do
                for CONV in ${CONVS[@]}; do
                    for FC in ${FCS[@]}; do
                        for PAT in ${PATIENCES[@]}; do
                            OUT_FILE="${OUT_DIR}/${KMERSTR}mer_${FILTER}f_${CONV}nc_${FC}fc_${PAT}pa_${EPOCHS}eps.txt"
                            FIG_DIR="${OUT_DIR}/figs/${KMERSTR}mer_${FILTER}f_${CONV}nc_${FC}fc_${PAT}pa_${EPOCHS}eps"
                            init_dir "$FIG_DIR"
                            export PY_ARGS="${FLAG} -ct -cv -i ${IN_DIR} -p ${POS_DIR} -s ${SEQ_DIR} -b ${BATCH} -e ${EPOCHS} -c ${CONT_DIR} -o ${OUT_FILE} -f ${FIG_DIR} -fs ${FILTER} -nc ${CONV} -nf ${FC} -pa ${PAT} -kf ${KFOLD} ${KMER}"
                            #echo $PY_ARGS
                            if [[ $FLAG == *-g* ]]; then
                                echo "Using GPU"
                                #echo "qsub $ARGS -v PY_ARGS -N t${KMERSTR}_${FC}fc_${CONV}c_${FILTER}f -e $STDERR_DIR -o $STDOUT_DIR ./run_train_multikmer_gpu.sh"
                                JOB_ID=`qsub $ARGS -v PY_ARGS -N t${KMERSTR}_${FC}fc_${CONV}c_${FILTER}f -e $STDERR_DIR -o $STDOUT_DIR ./run_train_multikmer_gpu.sh`
                            else
                                echo "Using CPU"
                                JOB_ID=`qsub $ARGS -v PY_ARGS -N train_${KMERSTR} -e $STDERR_DIR -o $STDOUT_DIR ./run_train_multikmer_cpu.sh`
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
        done
    done
done
