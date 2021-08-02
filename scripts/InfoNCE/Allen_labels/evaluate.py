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
from scripts.InfoNCE.Allen_labels import configFileAllen as configFile

plt.switch_backend('Agg')

rid_list = None#['055', '035', '085']
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

####################################
############ PARAMETERS ############
####################################
""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--epoch',  default='FI', help='Load model from the epoch specified')
arg_parser.add_argument('--l_reg_l1', default=None, help='lambda for the l1 registration loss')
arg_parser.add_argument('--l_reg_ncc',default=None, help='lambda for the ncc registration loss')
arg_parser.add_argument('--l_regsmooth', default=None, help='lambda for the registration smoothness loss')
arg_parser.add_argument('--l_GAN', default=None, help='lambda for the GAN loss')
arg_parser.add_argument('--l_nce', default=None, help='lambda for the  NCE loss')
arg_parser.add_argument('--temp', default=None, help='Temperature for the InfoNCE loss')

arguments = arg_parser.parse_args()
epoch_weights = str(arguments.epoch)
l_reg_l1 = arguments.l_reg_l1
l_reg_ncc = arguments.l_reg_ncc
l_regsmooth = arguments.l_regsmooth
l_GAN = arguments.l_GAN
l_nce = arguments.l_nce
temp = arguments.temp

parameter_dict = configFile.CONFIG

if l_reg_l1 is not None: parameter_dict['LAMBDA_REGISTRATION_L1'] = float(l_reg_l1)
if l_reg_ncc is not None: parameter_dict['LAMBDA_REGISTRATION_NCC'] = float(l_reg_ncc)
if l_regsmooth is not None: parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'] = float(l_regsmooth)
if l_GAN is not None: parameter_dict['LAMBDA_GAN'] = float(l_GAN)
if l_nce is not None: parameter_dict['LAMBDA_NCE'] = float(l_nce)
if temp is not None: parameter_dict['TEMPERATURE'] = float(temp)


parameter_dict['RESULTS_DIR'] += 'HISTO' + \
                                '_RL' + str(parameter_dict['LAMBDA_REGISTRATION_L1']) + \
                                '_RN' + str(parameter_dict['LAMBDA_REGISTRATION_NCC']) + \
                                '_RS' + str(parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS']) + \
                                '_N' + str(parameter_dict['LAMBDA_NCE']) + \
                                '_G' + str(parameter_dict['LAMBDA_GAN']) + \
                                '_T' + str(parameter_dict['TEMPERATURE']) 
kwargs_testing = {}
use_gpu = torch.cuda.is_available() and  parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")
###################################
########### DATA LOADER ###########
###################################
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
    landmarks=True
)



generator_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)

image_size_dict = {'M': dataset_test.image_shape, 'H': dataset_test.image_shape,
                   'latent': parameter_dict['N_CLASSES_NETWORK']}

#################################
############# MODEL #############
#################################
image_shape = dataset_test.image_shape

# Generator Network
G_M = models.ResnetGenerator(
    input_nc=input_channels_mri,
    output_nc=input_channels_histo,
    ngf=32,
    n_blocks=6,
    norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
    n_downsampling=3,
    device=device,
    tanh=True if 'TANH' in parameter_dict['PARENT_DIRECTORY'] else False
)

registration_M = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)

G_M = G_M.to(device)
registration_M = registration_M.to(device)


model_dict = {'G_M': G_M, 'R_M': registration_M}
epoch_results_dir = 'model_checkpoint.' + epoch_weights
if epoch_weights != 'None':
    weightsfile = 'model_checkpoint.' + str(epoch_weights) + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
    for model_key, model in model_dict.items():
        model.load_state_dict(checkpoint['state_dict_' + model_key])
        model.eval()

######################################
############# PREDICTION #############
######################################
modality_list = ['M', 'H']
results_dir = join(parameter_dict['RESULTS_DIR'], 'results', epoch_results_dir)
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

    output_results = evaluate_class.evaluate_rbs(data_dict, model_dict, device)
    H, seg_M, seg_H, rseg_H, landmarks = output_results
    init_mse, reg_mse = landmarks
    total_init_mse.append(np.mean(init_mse))
    total_fi_mse.append(np.mean(reg_mse))

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