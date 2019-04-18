#!/bin/bash

# This is an example script of training and running model ensembles.

# train 5 models with different seeds
python train.py --seed 5 --id 05 --save_dir saved_models/ensemble
python train.py --seed 6 --id 06 --save_dir saved_models/ensemble
python train.py --seed 7 --id 07 --save_dir saved_models/ensemble
python train.py --seed 8 --id 08 --save_dir saved_models/ensemble
python train.py --seed 9 --id 09 --save_dir saved_models/ensemble

# evaluate on test sets and save prediction files
python eval.py saved_models/ensemble/05 --out saved_models/ensemble/out/test_5.pkl
python eval.py saved_models/ensemble/06 --out saved_models/ensemble/out/test_6.pkl
python eval.py saved_models/ensemble/07 --out saved_models/ensemble/out/test_7.pkl
python eval.py saved_models/ensemble/08 --out saved_models/ensemble/out/test_8.pkl
python eval.py saved_models/ensemble/09 --out saved_models/ensemble/out/test_9.pkl

# run ensemble
ARGS=""
for id in 5 6 7 8 9; do
    OUT="saved_models/ensemble/out/test_${id}.pkl"
    ARGS="$ARGS $OUT"
done
python ensemble.py --dataset test $ARGS

