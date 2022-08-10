#!/bin/bash

set -u
export STDERR_DIR="./err"
export STDOUT_DIR="./out"
#init_dir "$STDERR_DIR" "$STDOUT_DIR"

#DIRS
CHAR_CORPUS="/rsgrps/bhurwitz/taxonomic_class/data/char_corpus"
CENT_DIR="/rsgrps/bhurwitz/taxonomic_class/data/CentrifugeSim"
BUG_DIR="/rsgrps/bhurwitz/taxonomic_class/data/BugMixingSim"
RESULTS_DIR="/rsgrps/bhurwitz/taxonomic_class/data/test_everything"

#PARAMETERS
WIND=10
CONT_SIZE=100
SEQ_E=10000
POS_E=10000
SEQ_HD=12
POS_HD=12
EPOCHS=100
BATCH=50
KMER_STEP="k"
NOSTOP=""

#TEST
#KMERS=(3)
#CONTIGS=("hiseq_100")
#ORGS=("ecoli-saprophyticus")
#TOKENS=("one_hot")


#LOOPERS
KMERS=(3)
CONTIGS=("hiseq_100")
#ORGS=("saprophyticus-pyogenes")
ORGS=("saprophyticus-pyogenes")
TOKENS=("one_hot")
FLAG="-up -uc -us -g -sc"
FILTERS=(3)
FCS=(2)
CONVS=(2)
PATIENCES=(5)

ARGS="-q standard -W group_list=bhurwitz -M mattmiller899@email.arizona.edu -m a"
for READ in ${CONTIGS[@]}; do
    for ORG in ${ORGS[@]}; do
        OUT_DIR="${RESULTS_DIR}/${READ}/${ORG}"
        #init_dir "$OUT_DIR" "$OUT_DIR/loss_figs"
        for KMER in ${KMERS[@]}; do
            IN_FILE="${BUG_DIR}/kmers/${READ}/${KMER}/${ORG}-reads.txt"
            CONT_EMB="${CENT_DIR}/embeddings/step_${KMER_STEP}/${READ}/${KMER}k_${WIND}w_${CONT_SIZE}s.txt"
            CONT_IN="cont_${WIND}cw_${CONT_SIZE}cs"
            if [ ! -f $CONT_EMB ] || [[ $FLAG != *-uc*  ]]; then
                echo "$CONT_EMB not found/used"
                CONT_IN="nocont"
            fi
            for TOKEN in ${TOKENS[@]}; do
                POS_EMB="${CHAR_CORPUS}/new_results/BP_pos_${KMER}mer_${TOKEN}_${POS_HD}hd_${POS_E}e_50batch_vectors.txt"
                POS_IN="pos_${POS_HD}phd_${POS_E}pe"
                if [ ! -f $POS_EMB ] || [[ $FLAG != *-up* ]]; then
                    echo "$POS_EMB not found/used"
                    POS_IN="nopos"
                fi

                SEQ_EMB="${CHAR_CORPUS}/new_results/BP_seq_${KMER}mer_${TOKEN}_${SEQ_HD}hd_${SEQ_E}e_50batch_vectors.txt"
                SEQ_IN="seq_${SEQ_HD}shd_${SEQ_E}se"
                if [ ! -f $SEQ_EMB ] || [[ $FLAG != *-us* ]]; then
                    echo "$SEQ_EMB not found/used"
                    SEQ_IN="noseq"
                fi
                for FILTER in ${FILTERS[@]}; do
                    for CONV in ${CONVS[@]}; do
                        for FC in ${FCS[@]}; do
                            for PAT in ${PATIENCES[@]}; do
                                OUT_FILE="${OUT_DIR}/OG_${KMER}mer_${FILTER}f_${CONV}nc_${FC}fc_${PAT}pa_${CONT_IN}_${POS_IN}_${SEQ_IN}_${EPOCHS}eps.txt"
                                FIG_DIR="${OUT_DIR}/figs/OG_${KMER}mer_${FILTER}f_${CONV}nc_${FC}fc_${PAT}pa_${CONT_IN}_${POS_IN}_${SEQ_IN}_${EPOCHS}eps"
                                init_dir "$FIG_DIR"
                                export PY_ARGS="${FLAG} -k ${KMER} -ct -cv -i ${IN_FILE} -p ${POS_EMB} -s ${SEQ_EMB} -b ${BATCH} -e ${EPOCHS} -c ${CONT_EMB} -o ${OUT_FILE} -f ${FIG_DIR} -fs ${FILTER} -nc ${CONV} -nf ${FC} -pa ${PAT}"
                                #echo $PY_ARGS
                                if [[ $FLAG == *-g* ]]; then
                                    echo "Using GPU"
                                    JOB_ID=`qsub $ARGS -v PY_ARGS -N t${KMER}_${FC}fc_${CONV}c_${FILTER}f -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_gpu.sh`
                                else
                                    echo "Using CPU"
                                    JOB_ID=`qsub $ARGS -v PY_ARGS -N train_${KMER}_${TOKEN} -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_cpu.sh`
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
done
