import functools
from os.path import join, exists
from os import makedirs
from datetime import date, datetime
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from matplotlib import pyplot as plt, colors
from PIL import Image
import nibabel as nib

from src import models, datasets, testing
from src.utils import image_transforms as tf
from src.utils.image_utils import tanh2im
from src.utils.visualization import slices, plot_results
from scripts.InfoNCE.L2R import configFileL2R as configFile

plt.switch_backend('Agg')

date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

####################################
############ PARAMETERS ############
####################################
""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--epoch',  default='FI', help='Load model from the epoch specified')
arg_parser.add_argument('--stain', help='stain used', default='CT', choices=['CT',])
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
if l_reg_ncc is not None: parameter_dict['LAMBDA_REGISTRATION_LABELS'] = float(l_reg_ncc)
if l_regsmooth is not None: parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'] = float(l_regsmooth)
if l_GAN is not None: parameter_dict['LAMBDA_GAN'] = float(l_GAN)
if l_nce is not None: parameter_dict['LAMBDA_NCE'] = float(l_nce)
if temp is not None: parameter_dict['TEMPERATURE'] = float(temp)


parameter_dict['RESULTS_DIR'] += '_' + stain_flag + '_' + \
                                '_RL' + str(parameter_dict['LAMBDA_REGISTRATION_L1']) + \
                                '_RN' + str(parameter_dict['LAMBDA_REGISTRATION_NCC']) + \
                                '_RLab' + str(parameter_dict['LAMBDA_REGISTRATION_LABELS']) + \
                                '_RS' + str(parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS']) + \
                                '_N' + str(parameter_dict['LAMBDA_NCE']) + \
                                '_G' + str(parameter_dict['LAMBDA_GAN']) + \
                                '_T' + str(parameter_dict['TEMPERATURE']) 
kwargs_testing = {}
use_gpu = False#torch.cuda.is_available() and  parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")
try:
    plot_results(join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                 keys=['loss_registration_l1', 'loss_registration_smoothness', 'loss_nce',
                       'loss_G', 'loss_D', 'loss'], restrict_ylim=True)
except:
    pass
###################################
########### DATA LOADER ###########
###################################
input_channels_mri = 1
input_channels_histo = 1

db_loader = parameter_dict['DB_CONFIG']['DATA_LOADER']
data_loader = db_loader.DataLoader(parameter_dict['DB_CONFIG'])#, rid_list=['0002_tcia'])

dataset_test = datasets.InterModalRegistrationDataset3D(
    data_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    ref_modality='MRI',
    flo_modality='HISTO',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7, 7)),
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
G_M = models.ResnetGenerator3D(
    input_nc=1,
    output_nc=1,
    ngf=6,
    n_blocks=2,
    norm_layer=functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False),
    n_downsampling=3,
    device=device,
    tanh=True if 'TANH' in parameter_dict['PARENT_DIRECTORY'] else False
)

registration_M = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7 if parameter_dict['FIELD_TYPE'] == 'velocity' else 0,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    device=device
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
    output_results = testing.predict_batch(data_dict, model_dict, image_size_dict, device)
    raw_data, mask_data, gen_data, labels, flow = output_results

    if 'TANH' in parameter_dict['RESULTS_DIR']:
        data_M, data_H, reg_H, gen, gen_reg = tanh2im(raw_data + gen_data, mask_data + mask_data[1:])
    else:
        data_M, data_H, reg_H = raw_data
        mask_M, mask_H, reg_mask_H = mask_data
        gen, gen_reg = gen_data
        labels_M, reg_labels_H = labels

    data_M = np.clip(data_M, 0, 1) * 255
    data_H = np.clip(data_H, 0, 1) * 255
    reg_H = np.clip(reg_H, 0, 1) * 255
    gen = np.clip(gen, 0, 1) * 255
    gen_reg = np.clip(gen_reg, 0, 1) * 255

    #### Generated images

    img = nib.Nifti1Image(data_M.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'M_orig.nii.gz'))

    img = nib.Nifti1Image(data_H.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'H_orig.nii.gz'))

    img = nib.Nifti1Image(reg_H.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'H_reg.nii.gz'))

    img = nib.Nifti1Image(gen.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'M_fake.nii.gz'))

    img = nib.Nifti1Image(gen_reg.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'M_fake_reg.nii.gz'))

    img = nib.Nifti1Image(labels_M.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'M_labels.nii.gz'))

    img = nib.Nifti1Image(reg_labels_H.astype(np.uint8), subject.vox2ras)
    nib.save(img, join(results_sbj_dir, 'H_reg_labels.nii.gz'))


print('Done.')
# output_results = testing.predict(generator_test, model_dict, image_size_dict, device)
# raw_data, mask_data, gen_data, labels, flow = output_results
#
# for it_image, (rid, subject) in enumerate(data_loader.subject_dict.items()):
#     print(str(it_image) + '/' + str(len(dataset_test)) + '. Slice: ' + str(data_loader.rid_list[it_image]))
#     results_sbj_dir = join(results_dir, str(data_loader.rid_list[it_image]))
#
#     data_M = raw_data[0][it_image]
#     data_H = raw_data[1][it_image]
#     reg_H = raw_data[2][it_image]
#     mask_M = mask_data[0][it_image]
#     mask_H = mask_data[1][it_image]
#     reg_mask_H = mask_data[2][it_image]
#     gen = gen_data[0][it_image]
#     gen_reg = gen_data[1][it_image]
#     labels_M = labels[0][it_image]
#     reg_labels_H = labels[1][it_image]
#
#     if 'TANH' in parameter_dict['RESULTS_DIR']:
#         data_M, data_H, reg_H, gen, gen_reg = tanh2im([data_M, data_H, reg_H, gen, gen_reg],
#                                                       [mask_M, mask_H, reg_mask_H, mask_H, reg_mask_H])
#     data_M = np.clip(data_M, 0, 1) * 255
#     data_H = np.clip(data_H, 0, 1) * 255
#     reg_H = np.clip(reg_H, 0, 1) * 255
#     gen = np.clip(gen, 0, 1) * 255
#     gen_reg = np.clip(gen_reg, 0, 1) * 255
#
#     #### Generated images
#
#     img = nib.Nifti1Image(data_M.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'M_orig.nii.gz'))
#
#     img = nib.Nifti1Image(data_H.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'H_orig.nii.gz'))
#
#     img = nib.Nifti1Image(reg_H.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'H_reg.nii.gz'))
#
#     img = nib.Nifti1Image(gen.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'M_fake.nii.gz'))
#
#     img = nib.Nifti1Image(gen_reg.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'M_fake_reg.nii.gz'))
#
#     img = nib.Nifti1Image(labels_M.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'M_labels.nii.gz'))
#
#     img = nib.Nifti1Image(reg_labels_H.astype(np.uint8), subject.vox2ras)
#     nib.save(img, join(results_sbj_dir, 'H_reg_labels.nii.gz'))
#
