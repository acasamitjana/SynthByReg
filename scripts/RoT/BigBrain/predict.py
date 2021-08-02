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
from src.utils.image_utils import tanh2im
from src.utils.visualization import slices, plot_results
from scripts.InfoNCE.BigBrain import configFileBigBrain as configFile

plt.switch_backend('Agg')

rid_list = ["{:03d}".format(i) for i in range(20, 351, 50)]#["{:03d}".format(i) for i in range(20, 351)]
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

####################################
############ PARAMETERS ############
####################################
""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--epoch',  default='FI', help='Load model from the epoch specified')
arg_parser.add_argument('--l_reg_l1', default=None, help='lambda for the l1 registration loss')
arg_parser.add_argument('--l_regsmooth', default=None, help='lambda for the registration smoothness loss')
arg_parser.add_argument('--l_GAN', default=None, help='lambda for the GAN loss')

arguments = arg_parser.parse_args()
epoch_weights = str(arguments.epoch)
l_reg_l1 = arguments.l_reg_l1
l_regsmooth = arguments.l_regsmooth
l_GAN = arguments.l_GAN

parameter_dict = configFile.CONFIG_RoT

if l_reg_l1 is not None: parameter_dict['LAMBDA_REGISTRATION_L1'] = float(l_reg_l1)
if l_regsmooth is not None: parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'] = float(l_regsmooth)
if l_GAN is not None: parameter_dict['LAMBDA_GAN'] = float(l_GAN)

parameter_dict['RESULTS_DIR'] += 'HISTO_'+ \
                                '_RL' + str(parameter_dict['LAMBDA_REGISTRATION_L1']) + \
                                '_RS' + str(parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS']) + \
                                '_G' + str(parameter_dict['LAMBDA_GAN'])
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

for batch_idx, data_dict in enumerate(generator_test):
    rid = data_dict['rid'][0]
    print(str(batch_idx) + '/' + str(len(dataset_test)) + '. Slice: ' + str(rid))

    subject = data_loader.subject_dict[rid]
    results_sbj_dir = join(results_dir, str(rid))
    output_results = testing.predict_batch_RoT(data_dict, model_dict, image_size_dict, device)
    raw_data, mask_data, gen_data, labels, flow = output_results


    #### Flow
    slices_2d = [flow[0], flow[1]]
    titles = ['FLOW_X image', 'FLOW_Y image']
    slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True, show=False)
    plt.savefig(join(results_sbj_dir, 'flow.png'))
    plt.close()

    if 'TANH' in parameter_dict['RESULTS_DIR']:
        data_M, data_H, reg_H, gen, gen_reg = tanh2im(raw_data + gen_data, mask_data + mask_data[1:])

    else:
        data_M, data_H, reg_H = raw_data
        mask_M, mask_H, reg_mask_H = mask_data
        gen, gen_reg = gen_data

    data_M = np.clip(data_M, 0, 1) * 255
    data_H = np.clip(data_H, 0, 1) * 255
    reg_H = np.clip(reg_H, 0, 1) * 255
    gen = np.clip(gen, 0, 1) * 255
    gen_reg = np.clip(gen_reg, 0, 1) * 255

    #### Generated images
    img = Image.fromarray(data_M.astype(np.uint8), mode='L')
    img.save(join(results_sbj_dir, 'M_orig.png'))

    img = Image.fromarray(data_H.astype(np.uint8), mode='L')
    img.save(join(results_sbj_dir, 'H_orig.png'))

    img = Image.fromarray(reg_H.astype(np.uint8), mode='L')
    img.save(join(results_sbj_dir, 'H_reg.png'))

    img = Image.fromarray(gen.astype(np.uint8), mode='L')
    img.save(join(results_sbj_dir, 'M_fake.png'))

    img = Image.fromarray(gen_reg.astype(np.uint8), mode='L')
    img.save(join(results_sbj_dir, 'M_fake_reg.png'))

print('Done.')
