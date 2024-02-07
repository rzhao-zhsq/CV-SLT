## CV-SLT

This repo holds codes of the [paper](https://arxiv.org/abs/2312.15645): Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment.

The CV-SLT builds upon the strong baseline [MMTLB](https://arxiv.org/abs/2203.04287), many thanks to their great work!

## Introduction

We propose CV-SLT to facilitate direct and sufficient cross-modal alignment between sign language videos and spoken language text. Specifically, our CV-SLT consists of two paths with two KL divergences to regularize the outputs of the encoder and decoder, respectively. In the *prior path*, the model solely relies on visual information to predict the target text; whereas in the *posterior path*, it simultaneously encodes visual information and textual knowledge to reconstruct the target text. Experiments conducted on public datasets (PHOENIX14T and CSL-daily) demonstrate the effectiveness of our framework, achieving new state-of-the-art results while significantly alleviating the cross-modal representation discrepancy. 

![Detailed model framework of CV-SLT](./figs/model.jpg)

## Performance

| Dataset    | R (Dev) | B1    | B2    | B3    | B4    | R (Test) | B1    | B2    | B3    | B4    |
| ---------- | ------- | ----- | ----- | ----- | ----- | -------- | ----- | ----- | ----- | ----- |
| PHOENIX14T | 55.05   | 55.35 | 42.99 | 35.07 | 29.55 | 54.54    | 54.76 | 42.80 | 34.97 | 29.52 |
| CSL-daily  | 56.36   | 58.05 | 44.73 | 35.14 | 28.24 | 57.06    | 58.29 | 45.15 | 35.77 | 28.94 |

## Implementation

- The implementation for the *prior path* and the *posterior path* is given in  `./modeling/translation`

- The Gaussian Network equipped with ARGD is given in `./modeling/gaussian_net`

### Prerequisites 

```sh
conda env create -f environment.yml
conda activate slt
```

### Data preparation

The raw data are from:

- [PHOENIX14T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [CSL-daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

Please refer to the [implementation of MMTLB](https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/SingleStream-SLT.md)  for preparing the data and models, as CV-SLT simply focuses on the SLT training. Specifically, the required processed data and pre-trained models include:

- Pre-extracted visual features for [PPHENIX14T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EndgQUATcNRCj0pTKPNMA_kBxSE9iJSONqj1zq1kQAAn5g?e=BgbJCK) and [CSL-daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EjbL5fTAZbxOmGA5x7px8s8BbyJ4ml5e5TROB-GEWPXeBQ?e=Ks7GfH). Please download and place them under `./experiment`
- Pre-trained Visual Embedding (trained on s2g task) and mBart modules (trained on g2t task) following [MMTLB](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EuJlnAhX7h9NnvFZhQH-_fcBtV8lbnj2CphiuidhhcU69w?e=eOsQ4B). Please download the corresponding directories and place them under `./pretrained_models` 

> Note that the path is configured in the \*.yaml file and you can change it anywhere you want.
>
> We backup the ckpts used in this repo [here](https://1drv.ms/f/s!Alt7L3J6LlN7nQuEc33dZokyvArb?e=SQIvQX). 

### Train and Evaluate

**Train**

```
dataset=phoenix-2014t #phoenix14t / csl-daily
python -m torch.distributed.launch \
--nproc_per_node 1 \
--use_env training.py \
--config experiments/configs/SingleStream/${dataset}_vs2t.yaml
```

**Evaluate**

Upon finishing training, your can evaluate the model with:

```
dataset=phoenix-2014t #phoenix14t / csl-daily
python -m torch.distributed.launch \
--nproc_per_node 1 \
--use_env prediction.py  \
--config experiments/configs/SingleStream/${dataset}_vs2t.yaml
```

You can also reproduce our reported performance with our trained ckpts.

- [phoenix-2014t_vs2t](https://1drv.ms/f/s!Alt7L3J6LlN7nQ3YlGmt4I8f9bIw?e=4knQU1)  
- [csl-daily_vs2t](https://1drv.ms/f/s!Alt7L3J6LlN7nQ-k7S7e6LOl3Sll?e=x9gZEu)  

We also provide a trained g2t ckpt of CSL-daily to help re-train our CV-SLT since it is lost in the repo of MMTLB.

- [csl-daily_g2t](https://1drv.ms/f/s!Alt7L3J6LlN7nRHnpATRKLTFMMXN?e=rpa2CB), the blue scores are `35.24/34.70` on Dev/Test sets.

## TODO

- Clean and release the codes. &#x2714;
- Prepare and release the pre-trained ckpts. &#x2714;

## Citation

```
@InProceedings{
    Zhao_2024_AAAI,
    author    = {Rui Zhao, Liang Zhang, Biao Fu, Cong Hu, Jinsong Su, Yidong Chen},
    title     = {Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    year      = {2024},
}
```

