#!/usr/bin/env bash

#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.1 --temp 0.05 --cycle "reg"
#python train_noGAN.py --l_reg_l1 1 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.02 --temp 0.05 --cycle "reg"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.1 --temp 0.05 --cycle "reg"
python train.py --l_reg_l1 1 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0.02 --temp 0.05 --cycle "reg"
python train_noGAN.py --l_reg_l1 1 --l_reg_ncc 0 --l_regsmooth 0 --l_nce 0 --temp 0 --cycle "reg"

#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "reg_only_2"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.2 --temp 0.05 --cycle "none"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.5 --temp 0.05 --cycle "none"


#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.01 --temp 0.05 --cycle "idt"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "reg"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "idt"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "reg"

#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.01 --temp 0.05 --cycle "idt"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "reg"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "idt"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "reg"
#
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "idt"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "reg"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "idt"
#python train_noGAN.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "reg"
#
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "idt"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 1 --l_nce 0.1 --temp 0.05 --cycle "reg"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "idt"
#python train.py --l_reg_l1 5 --l_reg_ncc 0 --l_regsmooth 0.1 --l_nce 0.1 --temp 0.05 --cycle "reg"