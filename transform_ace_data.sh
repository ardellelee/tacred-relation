#!/bin/bash

# This is an example script of transforming ACE-style data.

mkdir dataset/ace05/

train_in="../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.train.nlp.dual.json"
train_out='dataset/ace05/train.json'

dev_in="../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.validation.nlp.dual.json"
dev_out='dataset/ace05/dev.json'

test_in="../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.test.nlp.dual.json"
test_out='dataset/ace05/test.json'


# 'stat' or 'write'
mode='write'
echo "==> Training data."
python -m data.transform_ace_data --input $train_in --output $train_out --mode $mode

echo "==> Dev data."
python -m data.transform_ace_data --input $dev_in --output $dev_out --mode $mode

echo "==> Test data."
python -m data.transform_ace_data --input $test_in --output $test_out --mode $mode

echo "==> Done."