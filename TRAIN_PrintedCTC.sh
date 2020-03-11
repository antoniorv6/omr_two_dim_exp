#!/bin/bash
PATHS=""
PATHS="${PATHS} CameraPrimusFolds/Fold1"
PATHS="${PATHS} CameraPrimusFolds/Fold2"
PATHS="${PATHS} CameraPrimusFolds/Fold3"
PATHS="${PATHS} CameraPrimusFolds/Fold4"
PATHS="${PATHS} CameraPrimusFolds/Fold5"

declare -i fold=1
declare -i foldss=1

for NAME in ${PATHS}; do
    python -u trainCTCPrinted.py -data_path=${NAME} -fold=${fold}
    fold=$((fold + 1))
done

for NAME in ${PATHS}; do
    python -u trainCTCPrinted3.py -data_path=${NAME} -fold=${foldss}
    foldss=$((foldss + 1))
done
