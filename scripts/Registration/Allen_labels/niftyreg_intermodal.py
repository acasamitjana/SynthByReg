import functools
from os.path import join, exists
from os import makedirs
from datetime import date, datetime
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image

from src import models, datasets, metrics, evaluate as evaluate_class
from src.utils.image_utils import one_hot_encoding
from src.utils.visualization import slices, plot_results
from scripts.Registration.Allen_labels import configFileAllen as configFile

plt.switch_backend('Agg')

rid_list = None
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

####################################
############ PARAMETERS ############
####################################
parameter_dict = configFile.CONFIG_INTERMODAL_NR

kwargs_testing = {}
use_gpu = torch.cuda.is_available() and  parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")

###############
# Data loader #
###############

input_channels_mri = 1
input_channels_histo = 1
nonlinear_field_size = [9, 9]

db_loader = parameter_dict['DB_CONFIG']['DATA_LOADER']
data_loader = db_loader.DataLoader(parameter_dict['DB_CONFIG'], rid_list=rid_list)

dataset_test = datasets.InterModalRegistrationDataset2D(
    data_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    ref_modality='MRI',
    flo_modality='HISTO',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7)),
    train=True,
    landmarks=True,
    to_tensor=False
)

generator_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)

image_size_dict = {'M': dataset_test.image_shape, 'H': dataset_test.image_shape}


######################################
############# PREDICTION #############
######################################
results_dir = parameter_dict['RESULTS_DIR']
for it_image in range(len(dataset_test)):
    if not exists(join(results_dir, str(data_loader.rid_list[it_image]))):
        makedirs(join(results_dir, str(data_loader.rid_list[it_image])))

total_init_mse = []
total_fi_mse = []

total_init_dice_0 = []
total_init_dice_1 = []
total_init_dice_2 = []
total_init_dice_3 = []

total_fi_dice_0 = []
total_fi_dice_1 = []
total_fi_dice_2 = []
total_fi_dice_3 = []

for batch_idx, data_dict in enumerate(generator_test):

    print(str(batch_idx) + '/' + str(len(generator_test)) + '. Slice: ' + str(data_dict['rid'][0]))
    results_sbj_dir = join(results_dir, str(data_dict['rid'][0]))

    output_results = evaluate_class.evaluate_niftyreg_register(data_dict)
    H, seg_M, seg_H, rseg_H, landmarks, flow = output_results
    init_mse, reg_mse = landmarks
    total_init_mse.append(np.mean(init_mse))
    total_fi_mse.append(np.mean(reg_mse))

    np.save(join(results_sbj_dir, 'flow.npy'))

    img = Image.fromarray((255 * H).astype(np.uint8), mode='RGB')
    img.save(join(results_sbj_dir, 'H_landmarks.png'))

    seg_M = one_hot_encoding(seg_M, parameter_dict['N_CLASSES_SEGMENTATION'])
    img = Image.fromarray((255 * np.transpose(seg_M[1:], axes=[1,2,0])).astype(np.uint8), mode='RGB')
    img.save(join(results_sbj_dir, 'M_seg.png'))

    seg_H = one_hot_encoding(seg_H, parameter_dict['N_CLASSES_SEGMENTATION'])
    img = Image.fromarray((255 * np.transpose(seg_H[1:], axes=[1,2,0])).astype(np.uint8), mode='RGB')
    img.save(join(results_sbj_dir, 'H_seg.png'))

    rseg_H = one_hot_encoding(rseg_H, parameter_dict['N_CLASSES_SEGMENTATION'])
    img = Image.fromarray((255 * np.transpose(rseg_H[1:], axes=[1,2,0])).astype(np.uint8), mode='RGB')
    img.save(join(results_sbj_dir, 'H_reg_seg.png'))

    dice_0 = metrics.dice_coefficient(seg_M[0], seg_H[0])
    total_init_dice_0.append(dice_0)
    dice_0 = metrics.dice_coefficient(seg_M[0], rseg_H[0])
    total_fi_dice_0.append(dice_0)
    dice_1 = metrics.dice_coefficient(seg_M[1], seg_H[1])
    dice_1 = 0 if np.isnan(dice_1) else dice_1
    total_init_dice_1.append(dice_1)
    dice_1 = metrics.dice_coefficient(seg_M[1], rseg_H[1])
    dice_1 = 0 if np.isnan(dice_1) else dice_1
    total_fi_dice_1.append(dice_1)

    dice_2 = metrics.dice_coefficient(seg_M[2], seg_H[2])
    total_init_dice_2.append(dice_2)
    dice_2 = metrics.dice_coefficient(seg_M[2], rseg_H[2])
    total_fi_dice_2.append(dice_2)
    dice_3 = metrics.dice_coefficient(seg_M[3], seg_H[3])
    total_init_dice_3.append(dice_3)
    dice_3 = metrics.dice_coefficient(seg_M[3], rseg_H[3])
    total_fi_dice_3.append(dice_3)


plt.figure()
plt.plot(total_init_mse, 'r', marker = '*', label='Initial')
plt.plot(total_fi_mse, 'b', marker='*', label='RegBySynth')
plt.grid()
plt.legend()
plt.title('MSE Init: ' + str(np.round(np.mean(total_init_mse),2)) + '. MSE RegBySynth: ' + str(np.round(np.mean(total_fi_mse),2)))
plt.savefig(join(results_dir, 'landmark_results.png'))

plt.figure()
plt.plot(total_init_dice_0, 'r', marker = '*', label='Initial')
plt.plot(total_fi_dice_0, 'b', marker='*', label='RegBySynth')
plt.grid()
plt.legend()
plt.title('Dice Bkg Init: ' + str(np.round(np.mean(total_init_dice_0),2)) + '. Dice Bkg RegBySynth: ' + str(np.round(np.mean(total_fi_dice_0),2)))
plt.savefig(join(results_dir, 'dice_0.png'))

plt.figure()
plt.plot(total_init_dice_1, 'r', marker = '*', label='Initial')
plt.plot(total_fi_dice_1, 'b', marker='*', label='RegBySynth')
plt.grid()
plt.legend()
plt.title('GM_LL Init: ' + str(np.round(np.mean([m for m in total_init_dice_1 if m>0]),2)) + '. Dice GM_LL RegBySynth: ' + str(np.round(np.mean([m for m in total_fi_dice_1 if m>0]),2)))
plt.savefig(join(results_dir, 'dice_1.png'))

plt.figure()
plt.plot(total_init_dice_2, 'r', marker = '*', label='Initial')
plt.plot(total_fi_dice_2, 'b', marker='*', label='RegBySynth')
plt.grid()
plt.legend()
plt.title('Dice WM Init: ' + str(np.round(np.mean(total_init_dice_2),2)) + '. Dice WM RegBySynth: ' + str(np.round(np.mean(total_fi_dice_2),2)))
plt.savefig(join(results_dir, 'dice_2.png'))

plt.figure()
plt.plot(total_init_dice_3, 'r', marker = '*', label='Initial')
plt.plot(total_fi_dice_3, 'b', marker='*', label='RegBySynth')
plt.grid()
plt.legend()
plt.title('Dice GM Init: ' + str(np.round(np.mean(total_init_dice_3),2)) + '. Dice GM RegBySynth: ' + str(np.round(np.mean(total_fi_dice_3),2)))
plt.savefig(join(results_dir, 'dice_3.png'))
print('Done.')