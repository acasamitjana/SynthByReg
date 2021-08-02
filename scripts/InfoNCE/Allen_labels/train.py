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
from src.training import InfoNCE2D
from scripts.InfoNCE.Allen_labels import configFileAllen

####################################
######## GLOBAL  PARAMETERS ########
####################################
DEBUG_FLAG = False
debugWriter = DebugWriter(DEBUG_FLAG)
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

mask_flag = True
clip_grad = False
num_patches = 256
num_nc = 256
nce_layers = [0, 4, 8, 12, 16, 20]


""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--db', help='dataset used', default='Allen')
arg_parser.add_argument('--l_reg_l1', default=None, help='lambda for the l1 registration loss')
arg_parser.add_argument('--l_reg_ncc',default=None, help='lambda for the ncc registration loss')
arg_parser.add_argument('--l_regsmooth', default=None, help='lambda for the registration smoothness loss')
arg_parser.add_argument('--l_GAN', default=None, help='lambda for the GAN loss')
arg_parser.add_argument('--l_nce', default=None, help='lambda for the  NCE loss')
arg_parser.add_argument('--temp', default=None, help='Temperature for the InfoNCE loss')
arg_parser.add_argument('--cycle', default='reg', help='Cycle type')

arguments = arg_parser.parse_args()
db_flag = arguments.db
l_reg_l1 = arguments.l_reg_l1
l_reg_ncc = arguments.l_reg_ncc
l_regsmooth = arguments.l_regsmooth
l_GAN = arguments.l_GAN
l_nce = arguments.l_nce
temp = arguments.temp
cycle = arguments.cycle

rid_list = None
configFile = configFileAllen
parameter_dict = configFile.CONFIG
parameter_dict['mask_flag'] = mask_flag
parameter_dict['clip_grad'] = clip_grad
parameter_dict['num_patches'] = num_patches
parameter_dict['num_nc'] = num_nc
parameter_dict['nce_layers'] = nce_layers

if l_reg_l1 is not None: parameter_dict['LAMBDA_REGISTRATION_L1'] = float(l_reg_l1)
if l_reg_ncc is not None: parameter_dict['LAMBDA_REGISTRATION_NCC'] = float(l_reg_ncc)
if l_regsmooth is not None: parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'] = float(l_regsmooth)
if l_GAN is not None: parameter_dict['LAMBDA_GAN'] = float(l_GAN)
if l_nce is not None: parameter_dict['LAMBDA_NCE'] = float(l_nce)
if temp is not None: parameter_dict['TEMPERATURE'] = float(temp)


parameter_dict['RESULTS_DIR'] += 'HISTO_' + cycle + '_' + \
                                '_RL' + str(parameter_dict['LAMBDA_REGISTRATION_L1']) + \
                                '_RN' + str(parameter_dict['LAMBDA_REGISTRATION_NCC']) + \
                                '_RS' + str(parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS']) + \
                                '_N' + str(parameter_dict['LAMBDA_NCE']) + \
                                '_G' + str(parameter_dict['LAMBDA_GAN']) + \
                                '_T' + str(parameter_dict['TEMPERATURE'])

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

experimentWriter.write(  'CUDA available: ' + str(torch.cuda.is_available()) +
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

# Feature extractor Network
F_M = models.PatchSampleF(
    use_mlp=True, device=device, nc=num_nc
)

registration_M = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=dataset_train.image_shape,
    int_steps=7 if parameter_dict['FIELD_TYPE'] == 'velocity' else 0,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)
registration_M = registration_M.to(device)
if 'TANH' in parameter_dict['PARENT_DIRECTORY']:
    checkpoint = torch.load(parameter_dict['WEIGHTS_REGISTRATION_MRI_TANH'], map_location=device)

elif 'SIGMOID' in parameter_dict['PARENT_DIRECTORY']:
    checkpoint = torch.load(parameter_dict['WEIGHTS_REGISTRATION_MRI_SIGMOID'], map_location=device)

else:
    raise ValueError("Specify a valid parent directory")
registration_M.load_state_dict(checkpoint['state_dict'])
for param in registration_M.parameters():
    param.requires_grad = False
registration_M.eval()

# Discriminator
discriminator_M = models.NLayerDiscriminator(
    input_nc=1, ndf=32, n_layers=3,
    norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
    device=device, init_type='xavier'
)

model_dict = {'G_M': G_M, 'D_M': discriminator_M, 'F_M': F_M, 'R_M': registration_M}

da_model = TensorDeformation(image_shape, parameter_dict['NONLINEAR'].lowres_size, device)

# Losses
loss_function_gan = losses.DICT_LOSSES[parameter_dict['LOSS_GAN']]
loss_function_reg_ncc = losses.NCC_Loss
loss_function_reg_l1 = losses.L1_Loss
loss_function_regs = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']]
loss_function_nce = []

loss_function_gan = loss_function_gan(name='gan', device=device)
loss_function_reg_ncc = loss_function_reg_ncc(name='registration_ncc',device=device, kernel_var=[5,5])
loss_function_reg_l1 = loss_function_reg_l1(name='registration_l1')
loss_function_regs = loss_function_regs(dim=2, penalty='l2', name='registration_smoothness')
for nce_layer in nce_layers:
    loss_function_nce.append(losses.PatchNCELoss(batch_size=1, nce_T=parameter_dict['TEMPERATURE'],
                                                 name='NCE' + str(nce_layer)).to(device))

loss_function_dict = {
    'gan': loss_function_gan,
    'nce': loss_function_nce,
    'registration_ncc': loss_function_reg_ncc,
    'registration_l1': loss_function_reg_l1,
    'registration_smoothness': loss_function_regs,
}

lambda_weight_dict = {
    'gan': parameter_dict['LAMBDA_GAN'],
    'nce': parameter_dict['LAMBDA_NCE'],
    'registration_ncc': parameter_dict['LAMBDA_REGISTRATION_NCC'],
    'registration_l1': parameter_dict['LAMBDA_REGISTRATION_L1'],
    'registration_smoothness': parameter_dict['LAMBDA_REGISTRATION_SMOOTHNESS'],
}

# Set learning rates
gen_lr = parameter_dict['LEARNING_RATE']['generator']
dx_lr = parameter_dict['LEARNING_RATE']['discriminator']

optimizer_G_M = torch.optim.Adam(G_M.parameters(), lr=gen_lr, betas=(0.5, 0.999))
optimizer_D_M = torch.optim.Adam(discriminator_M.parameters(), lr=dx_lr, betas=(0.5, 0.999))
optimizer_dict = {'G_M': optimizer_G_M, 'D_M': optimizer_D_M}

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
        if optimizer_key == 'F_M':
            continue
        optimizer.load_state_dict(checkpoint['optimizer_' + optimizer_key])

    for model_key, model in model_dict.items():
        if model_key == 'F_M':
            continue
        model.load_state_dict(checkpoint['state_dict_' + model_key])

lrdecay = [LRDecay(o, n_iter_start=0, n_iter_finish=parameter_dict['N_EPOCHS']) for o in optimizer_dict.values()]
callback_list = []# + lrdecay

training_session = InfoNCE2D(device, loss_function_dict, lambda_weight_dict, callback_list, da_model, nce_layers,
                                 num_patches,  clip_grad, dx_lr, parameter_dict, cycle=cycle, mask_nce_flag=True)
training_session.train(model_dict, optimizer_dict, generator_train, **kwargs_training)