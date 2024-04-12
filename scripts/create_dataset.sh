#!/bin/bash

DATASET=Chair

python create_dataset.py --data_dir "path/to/prep_data" \
    --category $DATASET \
    --train_data_fn "${DATASET}.train.npy" \
    --val_data_fn "${DATASET}.val.npy" \
    --how_many_fusion -1 \
    --how_many_noise_data 0 \
    --data_how_many 2 3 \
    --data_how_many_prob 0.7 0.3 \
    --new_dataset_dir ./dataset/MixPartRandom \
    --train_data_epochs 2 \
    --test_data_epochs 2 \
    --num_workers 8 \
    --level 3

DATASET=Table

python create_dataset.py --data_dir "path/to/prep_data" \
    --category $DATASET \
    --train_data_fn "${DATASET}.train.npy" \
    --val_data_fn "${DATASET}.val.npy" \
    --how_many_fusion -1 \
    --how_many_noise_data 0 \
    --data_how_many 2 3 \
    --data_how_many_prob 0.7 0.3 \
    --new_dataset_dir ./dataset/MixPartRandom \
    --train_data_epochs 2 \
    --test_data_epochs 2 \
    --num_workers 8 \
    --level 3

DATASET=Lamp

python create_dataset.py --data_dir "path/to/prep_data" \
    --category $DATASET \
    --train_data_fn "${DATASET}.train.npy" \
    --val_data_fn "${DATASET}.val.npy" \
    --how_many_fusion -1 \
    --how_many_noise_data 0 \
    --data_how_many 2 3 \
    --data_how_many_prob 0.7 0.3 \
    --new_dataset_dir ./dataset/MixPartRandom \
    --train_data_epochs 3 \
    --test_data_epochs 3 \
    --num_workers 8 \
    --level 3
