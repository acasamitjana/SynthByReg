#!/usr/bin/env bash

python train_noGAN.py --stain "NISSL"
python train_noGAN.py --stain "IHC"
python train_noGAN.py --stain "Combined"
#python train_noGAN.py --stain "NISSL" --l_nce 0.05
#python train_noGAN.py --stain "NISSL" --l_nce 0.1

