# imports
from os.path import join
from datetime import date, datetime
from argparse import ArgumentParser

# third party imports
import numpy as np
import torch

# project imports
from src import losses, models, datasets
from src.utils.io import DebugWriter, ResultsWriter, create_results_dir, ExperimentWriter, worker_init_fn
from src.utils.tensor_utils import TensorDeformation
from src.callbacks import LRDecay
from src.training import Registration
from scripts.Registration.ERC import configFileERC as configFile

####################################
######## GLOBAL  PARAMETERS ########
####################################
DEBUG_FLAG = False
debugWriter = DebugWriter(DEBUG_FLAG)
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--stain', help='stain used', default='MRI', choices=['HE', 'LFB', 'MRI']),
arg_parser.add_argument('--block', help='BLOCK ID'),

arguments = arg_parser.parse_args()
stain_flag = arguments.stain
block = arguments.block

if stain_flag == 'LFB':
    parameter_dict = configFile.CONFIG_LFB
elif stain_flag == 'HE':
    parameter_dict = configFile.CONFIG_HE
else:
    parameter_dict = configFile.CONFIG

parameter_dict['RESULTS_DIR'] = join(parameter_dict['RESULTS_DIR'], block)
create_results_dir(parameter_dict['RESULTS_DIR'])
attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False

kwargs_training = {'log_interval': parameter_dict['LOG_INTERVAL']}  # Number of steps
device = torch.device("cuda:0" if parameter_dict['USE_GPU'] else "cpu")

debugWriter = DebugWriter(DEBUG_FLAG)
resultsWriter = ResultsWriter(join(parameter_dict['RESULTS_DIR'], 'experiment_parameters.txt'), attach=attach)
experimentWriter = ExperimentWriter(join(parameter_dict['RESULTS_DIR'], 'experiment.txt'), attach=attach)


resultsWriter.write('Experiment parameters\n')
for key, value in parameter_dict.items():
    resultsWriter.write(key + ': ' + str(value))
    resultsWriter.write('\n')
resultsWriter.write('\n')

###################################
########### DATA LOADER ###########
###################################
resultsWriter.write('Loading dataset ...\n')
DataLoader = parameter_dict['DB_CONFIG']['DATA_LOADER'].DataLoader
if stain_flag != 'MRI':
    data_loader = DataLoader(parameter_dict['DB_CONFIG'], stain_flag=stain_flag, bid_list=[block])
else:
    data_loader = DataLoader(parameter_dict['DB_CONFIG'], bid_list=[block])

block_loader = data_loader.subject_dict[block]
dataset_train = datasets.IntraModalRegistrationDataset2D(
    block_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    modality='MRI' if stain_flag == 'MRI' else 'HISTO',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7)),
    train=True,
    neighbor_distance=parameter_dict['DISTANCE_NEIGHBOURS']
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
model = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)
model.to(device)

da_model = TensorDeformation(image_shape,  parameter_dict['NONLINEAR'].lowres_size, device)

# Losses
loss_registration = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION']['name']]
loss_smoothness = losses.DICT_LOSSES[parameter_dict['LOSS_SMOOTHNESS']['name']]
loss_function_dict = {
    'registration':  loss_registration(device=device, name='registration', **parameter_dict['LOSS_REGISTRATION']['params']),
    'registration_smoothness': loss_smoothness(dim=2, penalty='l2', name='registration_smoothness',
                                               loss_mult=parameter_dict['UPSAMPLE_LEVELS'])
}
loss_weight_dict = {
    'registration':parameter_dict['LOSS_REGISTRATION']['lambda'],
    'registration_smoothness': parameter_dict['LOSS_SMOOTHNESS']['lambda']
}

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=parameter_dict['LEARNING_RATE'])


####################################
############# TRAINING #############
####################################

if parameter_dict['STARTING_EPOCH'] > 0:
    weightsfile = 'model_checkpoint.' + str(parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])

# Callbacks
lrdecay = LRDecay(optimizer, n_iter_start=0, n_iter_finish=parameter_dict['N_EPOCHS'])
callback_list = [lrdecay]# + lrdecay

training_session = Registration(device, loss_function_dict, loss_weight_dict, callback_list, da_model, parameter_dict)
training_session.train(model, optimizer, generator_train, **kwargs_training)


print('Done.')
