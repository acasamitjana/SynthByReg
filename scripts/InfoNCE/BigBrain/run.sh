#!/usr/bin/env bash

#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.1 --temp 0.05 --cycle "reg"
python train_noGAN.py --l_reg_l1 1 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.02 --temp 0.05 --cycle "reg"
python train.py --l_reg_l1 1 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.02 --temp 0.05 --cycle "reg"
python train_noGAN.py --l_reg_l1 1 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0 --temp 0 --cycle "reg"

