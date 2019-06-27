#!/bin/bash

# This is an example script of transforming ACE-style data.

mkdir dataset/ace03/

train_in="/home/yue/Projects/relation-extraction-ly-dev/data/ace03_json/ace2003.train.nlp.dual.json"
train_out='dataset/ace03/train.json'

dev_in="/home/yue/Projects/relation-extraction-ly-dev/data/ace03_json/ace2003.validation.nlp.dual.json"
dev_out='dataset/ace03/dev.json'

test_in="/home/yue/Projects/relation-extraction-ly-dev/data/ace03_json/ace2003.test.nlp.dual.json"
test_out='dataset/ace03/test.json'


# 'stat' or 'write'
mode='write'
echo "==> Training data."
python -m data.transform_ace_data --input $train_in --output $train_out --mode $mode

echo "==> Dev data."
python -m data.transform_ace_data --input $dev_in --output $dev_out --mode $mode

echo "==> Test data."
python -m data.transform_ace_data --input $test_in --output $test_out --mode $mode

echo "==> Done."
