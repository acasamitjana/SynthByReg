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
from src.training import WeaklyRegistration
from scripts.Registration.L2R import configFileL2R as configFile

####################################
######## GLOBAL  PARAMETERS ########
####################################
DEBUG_FLAG = False
debugWriter = DebugWriter(DEBUG_FLAG)
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--contrast', help='stain used', default='MRI', choices=['CT', 'MRI']),

arguments = arg_parser.parse_args()
stain_flag = arguments.contrast

if stain_flag == 'CT':
    parameter_dict = configFile.CONFIG_CT
elif stain_flag == 'MRI':
    parameter_dict = configFile.CONFIG_MRI
else:
    raise ValueError

parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_weakly'
create_results_dir(parameter_dict['RESULTS_DIR'])
attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False

kwargs_training = {'log_interval': parameter_dict['LOG_INTERVAL'], 'bidir_flag': False}  # Number of steps
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
data_loader = parameter_dict['DB_CONFIG']['DATA_LOADER'].DataLoader(parameter_dict['DB_CONFIG'])

dataset_train = datasets.IntraModalRegistrationDataset3D(
    data_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    modality='MRI' if stain_flag == 'MRI' else 'HISTO',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7, 7)),
    train=True,
    neighbor_distance=0
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
    int_steps=7 if parameter_dict['FIELD_TYPE'] == 'velocity' else 0,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    device=device
)
model.to(device)

da_model = TensorDeformation(image_shape, parameter_dict['NONLINEAR'].lowres_size, device)

# Losses
loss_registration = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION']['name']]
loss_registration_lab = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION_LABELS']['name']]
loss_registration_smooth = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['name']]
loss_function_dict = {
    'registration': loss_registration(device=device, name='registration',
                                      **parameter_dict['LOSS_REGISTRATION']['params']),

    'registration_labels': loss_registration_lab(device=device, name='registration_lab',
                                                 **parameter_dict['LOSS_REGISTRATION_LABELS']['params']),

    'registration_smoothness': loss_registration_smooth(name='registration_smoothness',
                                                        loss_mult=parameter_dict['UPSAMPLE_LEVELS'],
                                                        **parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['params'])
}
loss_weight_dict = {
    'registration': parameter_dict['LOSS_REGISTRATION']['lambda'],
    'registration_labels': parameter_dict['LOSS_REGISTRATION']['lambda'],
    'registration_smoothness': parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda']
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

training_session = WeaklyRegistration(device, loss_function_dict, loss_weight_dict, callback_list, da_model, parameter_dict)
training_session.train(model, optimizer, generator_train, **kwargs_training)


print('Done.')
