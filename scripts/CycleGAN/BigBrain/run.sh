#!/usr/bin/env bash

python train.py --l_reg 5 --l_cycle 1 --l_regsmooth 0.1
python train.py --l_reg 5 --l_cycle 5 --l_regsmooth 0.1
python train.py --l_reg 1 --l_cycle 5 --l_regsmooth 0.1

