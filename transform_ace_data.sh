#!/bin/bash

# This is an example script of transforming ACE-style data.
mkdir dataset/ace/

train_in="../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.train.nlp.dual.json"
train_out='dataset/ace/ace2005.511set.train.tacred.json'

dev_in="../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.validation.nlp.dual.json"
dev_out='dataset/ace/ace2005.511set.validation.tacred.json'

test_in="../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.test.nlp.dual.json"
test_out='dataset/ace/ace2005.511set.test.tacred.json'


# process ace data and save
python -m data.transform_ace_data --input $train_in --output $train_out
python -m data.transform_ace_data --input $dev_in --output $dev_out
python -m data.transform_ace_data --input $test_in --output $test_out

echo "==> Done."