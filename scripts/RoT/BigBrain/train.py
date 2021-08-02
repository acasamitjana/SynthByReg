from os.path import join
from datetime import date, datetime
import numpy as np
from argparse import ArgumentParser
import functools

import torch
from torch import nn

from src import losses, models, datasets
from src.callbacks import LRDecay
from src.utils.tensor_utils import TensorDeformation
from src.utils.io import DebugWriter, ResultsWriter, create_results_dir, ExperimentWriter, worker_init_fn
from src.training import RoT
from scripts.RoT.BigBrain import configFileBigBrain

####################################
######## GLOBAL  PARAMETERS ########
####################################
DEBUG_FLAG = False
debugWriter = DebugWriter(DEBUG_FLAG)
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

mask_flag = True
clip_grad = False


""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--db', help='dataset used', default='Allen')
arg_parser.add_argument('--l_reg_l1', default=None, help='lambda for the l1 registration loss')
arg_parser.add_argument('--l_regsmooth', default=None, help='lambda for the registration smoothness loss')
arg_parser.add_argument('--l_GAN', default=None, help='lambda for the GAN loss')

arguments = arg_parser.parse_args()
db_flag = arguments.db
l_reg_l1 = arguments.l_reg_l1
l_regsmooth = arguments.l_regsmooth
l_GAN = arguments.l_GAN

rid_list = None
configFile = configFileBigBrain
parameter_dict = configFile.CONFIG
parameter_dict['mask_flag'] = mask_flag
parameter_dict['clip_grad'] = clip_grad

if l_reg_l1 is not None: parameter_dict['LAMBDA_REGISTRATION_L1'] = float(l_reg_l1)
if l_regsmooth is not None: parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'] = float(l_regsmooth)
if l_GAN is not None: parameter_dict['LAMBDA_GAN'] = float(l_GAN)


parameter_dict['RESULTS_DIR'] += 'HISTO_'+ \
                                '_RL' + str(parameter_dict['LAMBDA_REGISTRATION_L1']) + \
                                '_RS' + str(parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS']) + \
                                '_G' + str(parameter_dict['LAMBDA_GAN'])

kwargs_training = {'log_interval': parameter_dict['LOG_INTERVAL'], 'weakly_flag': False}  # Number of steps
kwargs_testing = {}
use_gpu = torch.cuda.is_available() and parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")

create_results_dir(parameter_dict['RESULTS_DIR'])

attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False

debugWriter = DebugWriter(DEBUG_FLAG)
resultsWriter = ResultsWriter(join(parameter_dict['RESULTS_DIR'], 'experiment_parameters.txt'), attach=attach)
experimentWriter = ExperimentWriter(join(parameter_dict['RESULTS_DIR'], 'experiment.txt'), attach=attach)

resultsWriter.write('Experiment parameters\n')
for key, value in parameter_dict.items():
    resultsWriter.write(key + ': ' + str(value))
    resultsWriter.write('\n')
resultsWriter.write('\n')

experimentWriter.write('CUDA available: ' + str(torch.cuda.is_available()) +
                       '. Using CUDA: ' + str(parameter_dict['USE_GPU']))

###################################
########### DATA LOADER ###########
###################################
experimentWriter.write('Loading dataset ...\n')

input_channels_mri = 1
input_channels_histo = 1
nonlinear_field_size = [9, 9]

db_loader = parameter_dict['DB_CONFIG']['DATA_LOADER']
data_loader = db_loader.DataLoader(parameter_dict['DB_CONFIG'], rid_list=rid_list)

dataset_train = datasets.InterModalRegistrationDataset2D(
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

generator_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=parameter_dict['BATCH_SIZE'],
    shuffle=True,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=worker_init_fn
)

#################################
############# MODEL #############
#################################
experimentWriter.write('Loading model ...\n')
image_shape = dataset_train.image_shape

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
    inshape=dataset_train.image_shape,
    int_steps=7 if parameter_dict['FIELD_TYPE'] == 'velocity' else 0,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)
registration_M = registration_M.to(device)

# Discriminator
discriminator_M = models.NLayerDiscriminator(
    input_nc=1, ndf=32, n_layers=3,
    norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
    device=device, init_type='xavier'
)

model_dict = {'G_M': G_M, 'D_M': discriminator_M, 'R_M': registration_M}

da_model = TensorDeformation(image_shape, parameter_dict['NONLINEAR'].lowres_size, device)

# Losses
loss_function_gan = losses.DICT_LOSSES[parameter_dict['LOSS_GAN']]
loss_function_reg_ncc = losses.NCC_Loss
loss_function_reg_l1 = losses.L1_Loss
loss_function_regs = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']]
loss_function_nce = []

loss_function_gan_rt = loss_function_gan(name='gan_rt', device=device)
loss_function_gan_tr = loss_function_gan(name='gan_tr', device=device)
loss_function_reg_rt = loss_function_reg_l1(name='registration_rt')
loss_function_reg_tr = loss_function_reg_l1(name='registration_tr')
loss_function_regs = loss_function_regs(dim=2, penalty='l2', name='registration_smoothness')

loss_function_dict = {
    'gan_rt': loss_function_gan_rt,
    'gan_tr': loss_function_gan_tr,
    'registration_rt': loss_function_reg_rt,
    'registration_tr': loss_function_reg_tr,
    'registration_smoothness': loss_function_regs,
}

lambda_weight_dict = {
    'gan_rt': parameter_dict['LAMBDA_GAN'],
    'gan_tr': parameter_dict['LAMBDA_GAN'],
    'registration_rt': parameter_dict['LAMBDA_REGISTRATION_L1'],
    'registration_tr': parameter_dict['LAMBDA_REGISTRATION_L1'],
    'registration_smoothness': parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'],
}

# Set learning rates
gen_lr = parameter_dict['LEARNING_RATE']['generator']
dx_lr = parameter_dict['LEARNING_RATE']['discriminator']

optimizer_G_M = torch.optim.Adam(G_M.parameters(), lr=gen_lr, betas=(0.5, 0.999))
optimizer_D_M = torch.optim.Adam(discriminator_M.parameters(), lr=dx_lr, betas=(0.5, 0.999))
optimizer_R_M = torch.optim.Adam(registration_M.parameters(), lr=gen_lr, betas=(0.5, 0.999))
optimizer_dict = {'G_M': optimizer_G_M, 'D_M': optimizer_D_M, 'R_M': optimizer_R_M}

experimentWriter.write('Model ...\n')
for model_id, model in model_dict.items():
    experimentWriter.write(model_id + ': ' + str(type(model)))
    for name, param in model.named_parameters():
        if param.requires_grad:
            experimentWriter.write(name + '. Shape:' + str(torch.tensor(param.data.size()).numpy()))
            experimentWriter.write('\n')


####################################
############# TRAINING #############
####################################
experimentWriter.write('Training ...\n')
experimentWriter.write('Number of images = ' + str(len(data_loader)))

if parameter_dict['STARTING_EPOCH'] > 0:
    weightsfile = 'model_checkpoint.' + str(parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
    for optimizer_key, optimizer in optimizer_dict.items():
        optimizer.load_state_dict(checkpoint['optimizer_' + optimizer_key])

    for model_key, model in model_dict.items():
        model.load_state_dict(checkpoint['state_dict_' + model_key])

lrdecay = [LRDecay(o, n_iter_start=0, n_iter_finish=parameter_dict['N_EPOCHS']) for o in optimizer_dict.values()]
callback_list = []# + lrdecay

training_session = RoT(device, loss_function_dict, lambda_weight_dict, callback_list, da_model,
                       clip_grad, dx_lr, parameter_dict)
training_session.train(model_dict, optimizer_dict, generator_train, **kwargs_training)