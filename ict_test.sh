#!/bin/bash

mkdir -p ./OUTPUT/log/

NAME=ICT_BASE
TIME=EVAL-$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 \
    ict_eval.py \
    -name ${NAME} \
    --aug_test \
    2>&1 | tee  ./OUTPUT/log/${NAME}_${TIME}.log

