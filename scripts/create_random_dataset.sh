#!/bin/bash

python create_random_dataset.py --category Chair \
    --data_dir dataset/MixPartRandom \
    --save_data_dir dataset/Random_MixPartRandom

python create_random_dataset.py --category Table \
    --data_dir dataset/MixPartRandom \
    --save_data_dir dataset/Random_MixPartRandom

python create_random_dataset.py --category Lamp \
    --data_dir dataset/MixPartRandom \
    --save_data_dir dataset/Random_MixPartRandom
