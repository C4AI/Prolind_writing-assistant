#!/bin/bash

python prepare_dataset.py -i ../data/constituicao_parallel_por_yrl_train.xlsx -o ../data/generated/constituicao_train.csv
python prepare_dataset.py -i ../data/constituicao_parallel_por_yrl_test.xlsx -o ../data/generated/constituicao_test.csv

