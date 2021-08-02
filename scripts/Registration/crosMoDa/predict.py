# imports
from os.path import join, exists
from os import makedirs

# third party imports
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import nibabel as nib

# project imports
from src import datasets, models
from src.testing import predict_registration
from src.utils.io import create_results_dir
from src.utils.visualization import slices, plot_results
from src.utils.image_utils import tanh2im
from scripts.Registration.L2R import configFileL2R as configFile
from src.utils.tensor_utils import TensorDeformation

plt.switch_backend('Agg')

num_predictions = 20

####################################
############ PARAMETERS ############
####################################
""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--epoch', default='FI', help='Load model from the epoch specified')
arg_parser.add_argument('--contrast', help='stain used', default='MRI', choices=['CT', 'MRI']),
arg_parser.add_argument('--weakly', action='store_true')

arguments = arg_parser.parse_args()
stain_flag = arguments.contrast
weakly_flag = arguments.weakly
epoch_weights = str(arguments.epoch)

if stain_flag == 'CT':
    parameter_dict = configFile.CONFIG_CT
elif stain_flag == 'MRI':
    parameter_dict = configFile.CONFIG_MRI
else:
    raise ValueError

parameter_dict['BATCH_SIZE'] = 1
if weakly_flag:
    parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_weakly'

kwargs_testing = {}
kwargs_generator = {'num_workers': 1, 'pin_memory': True} if parameter_dict['USE_GPU'] else {}
use_gpu = torch.cuda.is_available() and  parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")

create_results_dir(parameter_dict['RESULTS_DIR'])
plot_results(join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
             keys=['loss_registration', 'loss_registration_smoothness', 'loss'])
###################################
########### DATA LOADER ###########
###################################
data_loader = parameter_dict['DB_CONFIG']['DATA_LOADER'].DataLoader(parameter_dict['DB_CONFIG'])

num_predictions = np.clip(num_predictions, 0, np.ceil(len(data_loader)))

dataset_test = datasets.IntraModalRegistrationDataset3D(
    data_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    modality='MRI' if stain_flag == 'MRI' else 'HISTO',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7, 7)),
    train=False,
    neighbor_distance=0
)

generator_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=parameter_dict['BATCH_SIZE'],
    shuffle=True,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)

#################################
############# MDOEL #############
#################################
image_shape = dataset_test.image_shape
model = models.VxmRigidDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    device=device
)
model.to(device)

epoch_results_dir = 'model_checkpoint.' + epoch_weights
weightsfile = 'model_checkpoint.' + epoch_weights + '.pth'
checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

da_model = TensorDeformation(image_shape, parameter_dict['NONLINEAR'].lowres_size, device)

results_dir = join(parameter_dict['RESULTS_DIR'], 'results', epoch_results_dir)
if not exists(results_dir):
    makedirs(results_dir)

print('Writing results')
for batch_idx, data_dict in enumerate(generator_test):
    if batch_idx*parameter_dict['BATCH_SIZE'] >= num_predictions and batch_idx > 0:
        break

    print(str(batch_idx*parameter_dict['BATCH_SIZE']) + '/' + str(int(num_predictions)))

    ref_rid_list = data_dict['rid']
    output_results = predict_registration(data_dict, model, device, da_model)

    for it_image, rid_image in enumerate(ref_rid_list):
        reg_rid = str(rid_image)
        ref_rid = reg_rid.split('_to_')[0]
        flo_rid = reg_rid.split('_to_')[1]
        print(reg_rid)
        ref, flo, reg_r, reg_f = tanh2im([o[it_image] for o in output_results[:4]],
                                         mask_list=[o[it_image] for o in output_results[4:8]])

        img = nib.Nifti1Image(ref, data_loader.subject_dict[ref_rid].vox2ras)
        nib.save(img, join(results_dir, 'ref_' + ref_rid + '.nii.gz'))

        img = nib.Nifti1Image(flo, data_loader.subject_dict[ref_rid].vox2ras)
        nib.save(img, join(results_dir, 'flo_' + ref_rid + '.nii.gz'))

        img = nib.Nifti1Image(reg_r, data_loader.subject_dict[ref_rid].vox2ras)
        nib.save(img, join(results_dir, 'reg_ref_' + ref_rid + '.nii.gz'))

        img = nib.Nifti1Image(reg_f, data_loader.subject_dict[ref_rid].vox2ras)
        nib.save(img, join(results_dir, 'reg_flo_' + ref_rid + '.nii.gz'))

        img = nib.Nifti1Image(np.transpose(np.squeeze(output_results[-1]),axes=[1,2,3,0]), data_loader.subject_dict[ref_rid].vox2ras)
        nib.save(img, join(results_dir, 'flow_' + ref_rid + '.nii.gz'))

print('Predicting done.')
