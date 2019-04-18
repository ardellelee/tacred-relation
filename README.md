Position-aware Attention RNN Model for Relation Extraction
=========================
**This development branch is for code adaptation to use this model for [ACE](https://www.ldc.upenn.edu/collaborations/past-projects/ace) relation extraction dataset.**


This repo contains the *PyTorch* code for paper [Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf).

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).




## Requirements

- Python 3 (tested on 3.6.2)
- PyTorch (tested on 1.0.0)
- unzip, wget (for downloading only)

## Preparation

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

If the ACE data does not meet the requirements of this model, use the following scripts to transform:
```
chmod +x transform_ace_data.sh; ./transform_ace_data.sh
```

The script supports two modes: 'stat' and 'write'.

- stat mode: count the statistics of the dataset
- write mode: transform the data and save as json file


Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/ace05 dataset/vocab_ace05 --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab_ace05`.

## Training

Train a position-aware attention RNN model with:
```
python train.py --data_dir dataset/ace05 --vocab_dir dataset/vocab_ace05 --id 00 --save_dir ./saved_models_ace05 --info "Position-aware attention model"
```

Use `--topn N` to finetune the top N word vectors only. The script will do the preprocessing automatically (word dropout, entity masking, etc.).

Train an LSTM model with:
```
python train.py --data_dir dataset/ace05 --vocab_dir dataset/vocab_ace05 --no-attn --id 01 --save_dir ./saved_models_ace05 --info "LSTM model"
```

Model checkpoints and logs will be saved to `./saved_models/00`.

## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).

## Ensemble

Please see the example script `ensemble.sh`.

## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
