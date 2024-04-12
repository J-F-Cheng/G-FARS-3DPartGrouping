#!/bin/bash
TIME=$(date "+%Y%m%d%H%M%S")
DATASET="Chair"
SAMPLER="PC_origin"
EXP_NAME="eval_gfars_${DATASET}_${TIME}_${SAMPLER}"

python eval_gfars.py \
    --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir dataset/MixPartRandom \
    --num_workers 8 \
    --device cuda:0 \
    --model_path pretrain/chair.pth \
    --sel_sampler $SAMPLER \
    --val_batch_size 16


TIME=$(date "+%Y%m%d%H%M%S")
DATASET="Table"
SAMPLER="PC_origin"
EXP_NAME="eval_gfars_${DATASET}_${TIME}_${SAMPLER}"

python eval_gfars.py \
    --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir dataset/MixPartRandom \
    --num_workers 8 \
    --device cuda:0 \
    --model_path pretrain/table.pth \
    --sel_sampler $SAMPLER \
    --val_batch_size 16

TIME=$(date "+%Y%m%d%H%M%S")
DATASET="Lamp"
SAMPLER="PC_origin"
EXP_NAME="eval_gfars_${DATASET}_${TIME}_${SAMPLER}"

python eval_gfars.py \
    --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir dataset/MixPartRandom \
    --num_workers 8 \
    --device cuda:0 \
    --model_path pretrain/lamp.pth \
    --sel_sampler $SAMPLER \
    --val_batch_size 16
