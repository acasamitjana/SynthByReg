#!/usr/bin/env bash

python InfoNCE/L2R/train.py --l_reg_l1 5 --l_nce 0.1 --l_regsmooth 0.1 --temp 0.07 --l_reg_labels 0
python InfoNCE/L2R/train.py --l_reg_l1 20 --l_nce 0.1 --l_regsmooth 0.1 --temp 0.07 --l_reg_labels 0
#python InfoNCE/L2R/train.py --l_reg_l1 40 --l_nce 0.1 --l_regsmooth 0.1 --temp 0.07 --l_reg_labels 0
#python InfoNCE/L2R/train_noGAN.py --l_reg_l1 20 --l_nce 0.1 --l_regsmooth 0.1 --temp 0.07 --l_reg_labels 0
#python InfoNCE/L2R/train_noGAN.py --l_reg_l1 5 --l_nce 0.01 --l_regsmooth 0.1 --temp 0.07 --l_reg_labels 0
#python InfoNCE/L2R/train_noGAN.py --l_reg_l1 5 --l_nce 0.1 --l_regsmooth 0.1 --temp 0.01 --l_reg_labels 0
#python InfoNCE/L2R/train_noGAN.py --l_reg_l1 10 --l_nce 0.2 --l_regsmooth 0.1 --temp 0.005 --l_reg_labels 0
#python InfoNCE/L2R/train_noGAN.py --l_reg_l1 10 --l_nce 0.5 --l_regsmooth 0.1 --temp 0.005 --l_reg_labels 0
#python InfoNCE/L2R/train_noGAN.py --l_reg_l1 5 --l_nce 0.1 --l_regsmooth 1 --temp 0.005 --l_reg_labels 0
