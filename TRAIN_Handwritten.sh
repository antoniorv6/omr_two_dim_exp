#!/bin/bash
PATHS=""
PATHS="${PATHS} HandwrittenFolds/Fold1"
PATHS="${PATHS} HandwrittenFolds/Fold2"
PATHS="${PATHS} HandwrittenFolds/Fold3"
PATHS="${PATHS} HandwrittenFolds/Fold4"
PATHS="${PATHS} HandwrittenFolds/Fold5"

declare -i fold=1

#for NAME in ${PATHS}; do
#    python -u trainCTCHandwritten.py -data_path=${NAME} -fold=${fold}
#    fold=$((fold + 1))
#done

#fold = 0

for NAME in ${PATHS}; do
    python -u trainCTCHandwritten3.py -data_path=${NAME} -fold=${fold}
    fold=$((fold + 1))
done
