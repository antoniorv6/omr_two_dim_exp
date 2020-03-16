#!/bin/bash
PATHS=""
PATHS="${PATHS} CameraPrimusFolds/Fold1"
PATHS="${PATHS} CameraPrimusFolds/Fold2"
PATHS="${PATHS} CameraPrimusFolds/Fold3"
PATHS="${PATHS} CameraPrimusFolds/Fold4"
PATHS="${PATHS} CameraPrimusFolds/Fold5"

PATHSHW=""
PATHSHW="${PATHSHW} HandwrittenFolds/Fold1"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold2"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold3"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold4"
PATHSHW="${PATHSHW} HandwrittenFolds/Fold5"

declare -i foldhw=1
declare -i fold=1
declare -i foldss=1

python -u trainCTCHandwritten3.py -data_path="HandwrittenFolds/Fold4" -fold=4

for NAME in ${PATHS}; do
    python -u trainCTCPrinted.py -data_path=${NAME} -fold=${fold}
    fold=$((fold + 1))
done

for NAME in ${PATHS}; do
    python -u trainCTCPrinted3.py -data_path=${NAME} -fold=${foldss}
    foldss=$((foldss + 1))
done
