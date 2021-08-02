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

from src import models, datasets, testing
from src.utils import image_transforms as tf
from src.utils.image_utils import tanh2im
from src.utils.visualization import slices, plot_results
from scripts.InfoNCE.Cricks import configFileCricks as configFile

plt.switch_backend('Agg')
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

####################################
############ PARAMETERS ############
####################################
""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--epoch',  default='FI', help='Load model from the epoch specified')
arg_parser.add_argument('--stain', help='stain used', default='NEUN', choices=['NEUN', 'DAPI', 'Combined'])
arg_parser.add_argument('--l_reg_l1', default=None, help='lambda for the l1 registration loss')
arg_parser.add_argument('--l_reg_ncc',default=None, help='lambda for the ncc registration loss')
arg_parser.add_argument('--l_regsmooth', default=None, help='lambda for the registration smoothness loss')
arg_parser.add_argument('--l_GAN', default=None, help='lambda for the GAN loss')
arg_parser.add_argument('--l_nce', default=None, help='lambda for the  NCE loss')
arg_parser.add_argument('--temp', default=None, help='Temperature for the InfoNCE loss')

arguments = arg_parser.parse_args()
epoch_weights = str(arguments.epoch)
stain_flag = arguments.stain
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


parameter_dict['RESULTS_DIR'] += '_' + stain_flag + '_' + \
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
db_loader = parameter_dict['DB_CONFIG']['DATA_LOADER']
data_loader = db_loader.DataLoader(parameter_dict['DB_CONFIG'], stain=stain_flag)

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
)

generator_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=5,
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
    input_nc=1,
    output_nc=1,
    ngf=32,
    n_blocks=6,
    norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
    n_downsampling=3,
    device=device
)

registration_M = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    gaussian_filter_flag=True
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

output_results = testing.predict(generator_test, model_dict, image_size_dict, device)
raw_data, mask_data, gen_data, flow = output_results

data_M, data_H, reg_H, gen, gen_reg = tanh2im(raw_data + gen_data, mask_data + mask_data[1:])

data_M = np.transpose(data_M, axes=[1, 2, 0])
data_H = np.transpose(data_H, axes=[1, 2, 0])
reg_H = np.transpose(reg_H, axes=[1, 2, 0])
gen = np.transpose(gen, axes=[1, 2, 0])
gen_reg = np.transpose(gen_reg, axes=[1, 2, 0])

import nibabel as nib
HISTO_THICKNESS = 0.2
BLOCK_res = 0.02

HISTO_AFFINE = np.zeros((4, 4))
HISTO_AFFINE[0, 1] = BLOCK_res
HISTO_AFFINE[2, 0] = -BLOCK_res
HISTO_AFFINE[1, 2] = HISTO_THICKNESS

img = nib.Nifti1Image(data_M, HISTO_AFFINE)
nib.save(img, join(results_dir, 'M_orig.nii.gz'))

img = nib.Nifti1Image(data_H, HISTO_AFFINE)
nib.save(img, join(results_dir, 'H_orig.nii.gz'))

img = nib.Nifti1Image(reg_H, HISTO_AFFINE)
nib.save(img, join(results_dir, 'H_reg.nii.gz'))

img = nib.Nifti1Image(gen, HISTO_AFFINE)
nib.save(img, join(results_dir, 'M_fake.nii.gz'))

img = nib.Nifti1Image(gen_reg, HISTO_AFFINE)
nib.save(img, join(results_dir, 'M_fake_reg.nii.gz'))

print('Done.')