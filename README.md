# Stereo Super Resolution with Transformer


## Installation

Our code is based on Ubuntu 22.04, pytorch 1.10.0, CUDA 12.4 (NVIDIA RTX 4090 24GB) and python 3.10.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create --file environment.yaml
conda activate lte
```


## Quick Start

### 1. Download pre-trained models.

Model|Download
:-:|:-:
EDSR-baseline-LTE|[Google Drive](https://drive.google.com/file/d/108-wQJOTR41JNn_2Q-5X4p07DvgrBNSB/view?usp=sharing)
EDSR-baseline-LTE+|[Google Drive](https://drive.google.com/file/d/1k_BWZWC4tvWA0WouViHAicdTg0pHBp-W/view?usp=sharing)
RDN-LTE|[Google Drive](https://drive.google.com/file/d/1fdj5cvSopIqFi74x9rofPP9O_2HfSp7K/view?usp=sharing)
SwinIR-LTE|[Google Drive](https://drive.google.com/file/d/1DnrL86pUKwRXNLOxoK_GJdrP6IZ3y9nH/view?usp=sharing)

Model|Download
:-:|:-:
SwinIR-MetaSR|[Google Drive](https://drive.google.com/file/d/1uG8PTLRdKxO4AaFh3SNl_nFcHqeGtu4y/view?usp=sharing)
SwinIR-LIIF|[Google Drive](https://drive.google.com/file/d/1TvXXvimq-FlVB6tOye5J3xTqPv_DRl_P/view?usp=sharing)


### 2. Reproduce experiments.

**Table 1: EDSR-baseline-LTE**

```bash ./scripts/test-div2k.sh ./save/edsr-baseline-lte.pth 0```

**Table 1: RDN-LTE**

```bash ./scripts/test-div2k.sh ./save/rdn-lte.pth 0```

**Table 1: SwinIR-LTE**

```bash ./scripts/test-div2k-swin.sh ./save/swinir-lte.pth 8 0```

**Table 2: RDN-LTE**

```bash ./scripts/test-benchmark.sh ./save/rdn-lte.pth 0```

**Table 2: SwinIR-LTE**

```bash ./scripts/test-benchmark-swin.sh ./save/swinir-lte.pth 8 0```


## Train & Test

###  **EDSR-baseline-LTE**

**Train**: `python train.py --config configs/train-div2k/train_edsr-baseline-lte.yaml --gpu 0`

**Test**: `python test.py --config configs/test/test-div2k-2.yaml --model save/_train_edsr-baseline-lte/epoch-last.pth --gpu 0`

### **EDSR-baseline-LTE+**

**Train**: `python train.py --config configs/train-div2k/train_edsr-baseline-lte-fast.yaml --gpu 0`

**Test**: `python test.py --config configs/test/test-fast-div2k-2.yaml --fast True --model save/_train_edsr-baseline-lte-fast/epoch-last.pth --gpu 0`

### **RDN-LTE**

**Train**: `python train.py --config configs/train-div2k/train_rdn-lte.yaml --gpu 0,1`

**Test**: `python test.py --config configs/test/test-div2k-2.yaml --model save/_train_rdn-lte/epoch-last.pth --gpu 0`

### **SwinIR-LTE**

**Train**: `python train.py --config configs/train-div2k/train_swinir-lte.yaml --gpu 0,1,2,3`

**Test**: `python test.py --config configs/test/test-div2k-2.yaml --model save/_train_swinir-lte/epoch-last.pth --window 8 --gpu 0`

### **Ours**

**Train**: `python train.py --config configs/train-AID/train_ssrtr-lte.yaml --name ssrtr-lte --tag 2024-04-15 --gpu 0`
`python train.py --config configs/train-stereo/pretrain/train_naf-lte.yaml --name naf-lte --tag pretrained --gpu 0`
`python train.py --config configs/train-stereo/finetune/train_naf-lte.yaml --name naf-lte --tag finetune --gpu 0` 

**Test**: `python test.py --config configs/test/test-flickr-2.yaml --model save/ssrtr-lte_2024-03-09/epoch-last.pth --gpu 0`





Model|Training time (# GPU)
:-:|:-:
EDSR-baseline-LTE|21h (1 GPU)
RDN-LTE|82h (2 GPU)
SwinIR-LTE|75h (4 GPU)

We use NVIDIA RTX 3090 24GB for training.


## Fourier Space

The script [Eval-Fourier-Feature-Space](https://github.com/jaewon-lee-b/lte/blob/main/Eval-Fourier-Feature-Space.ipynb) is used to generate the paper plots.


## Demo

`python demo.py --input ./demo/Urban100_img012x2.png --model save/edsr-baseline-lte.pth --scale 2 --output output.png --gpu 0`


## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif), [SwinIR](https://github.com/JingyunLiang/SwinIR), [LTE](https://github.com/jaewon-lee-b/lte.git). We thank the authors for sharing their codes.