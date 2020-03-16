#!/bin/bash
PATHSHW=""
PATHSHW="${PATHSHW} HandwrittenFolds/Fold1"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold2"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold3"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold4"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold5"

declare -i fold=1

#for NAME in ${PATHSHW}; do
#    python -u trainCTCHandwritten.py -data_path=${NAME} -fold=${fold}
#    fold=$((fold + 1))
#done

#fold = 0

for NAME in ${PATHSHW}; do
    python -u trainCTCHandwritten3.py -data_path=${NAME} -fold=${fold}
    fold=$((fold + 1))
done
