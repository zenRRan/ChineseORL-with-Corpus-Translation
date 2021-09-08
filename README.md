# Chinese-ORL-with-Corpus-Translation

This repository contains codes and configs for training models presented in: 

>Chinese Opinion Role Labeling with Corpus Translation: A Pivot Study [EMNLP2021]

## Train and Test
To train our ChineseORL-with-Corpus-Trainslation models, you should get into one folder what you want, and set your corpus in `expdata/opinion.cfg`

set your GPU device in `runme.sh`:
```
export CUDA_VISIBLE_DEVICES=[gpu_num]
```
run:
```
sh runme.sh
```
During the training process, test will conduct every n step, set n in `expdata/opinion.cfg`:
```
validate_every = [n]
```
In our training process, we set test every half epoch.