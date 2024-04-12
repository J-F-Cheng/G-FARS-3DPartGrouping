#!/bin/bash
TIME=$(date "+%Y%m%d%H%M%S")
DATASET="Chair"
EXP_NAME="train_gfars_${DATASET}_${TIME}"

python train_gfars.py --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir dataset/Random_MixPartRandom \
    --data_dir_test dataset/MixPartRandom \
    --num_workers 8 \
    --device cuda:0 \
    --epochs 3000 \
    --batch_size 16 \
    --epoch_save 25 \
    --val_every_epochs 25 \
    --val_how_many_gen 8 \
    --val_batch_size 4

TIME=$(date "+%Y%m%d%H%M%S")
DATASET="Table"
EXP_NAME="train_gfars_${DATASET}_${TIME}"

python train_gfars.py --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir dataset/Random_MixPartRandom \
    --data_dir_test dataset/MixPartRandom \
    --num_workers 8 \
    --device cuda:0 \
    --epochs 3000 \
    --batch_size 16 \
    --epoch_save 25 \
    --val_every_epochs 25 \
    --val_how_many_gen 8 \
    --val_batch_size 4

TIME=$(date "+%Y%m%d%H%M%S")
DATASET="Lamp"
EXP_NAME="train_gfars_${DATASET}_${TIME}"

python train_gfars.py --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir dataset/Random_MixPartRandom \
    --data_dir_test dataset/MixPartRandom \
    --num_workers 8 \
    --device cuda:0 \
    --epochs 3000 \
    --batch_size 16 \
    --epoch_save 25 \
    --val_every_epochs 25 \
    --val_how_many_gen 8 \
    --val_batch_size 4

