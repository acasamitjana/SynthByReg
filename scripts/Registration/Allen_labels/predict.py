# imports
from os.path import join, exists
from os import makedirs

# third party imports
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser

# project imports
from src import datasets, models
from src.testing import predict_registration
from src.utils.io import create_results_dir
from src.utils.visualization import slices, plot_results
from src.utils.image_utils import tanh2im
from scripts.Registration.Allen_labels import configFileAllen as configFile
from src.utils.tensor_utils import TensorDeformation

plt.switch_backend('Agg')
num_predictions = 20

####################################
############ PARAMETERS ############
####################################
""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--epoch', default='FI', help='Load model from the epoch specified')

arguments = arg_parser.parse_args()
epoch_weights = str(arguments.epoch)

parameter_dict = configFile.CONFIG
parameter_dict['BATCH_SIZE'] = 1

kwargs_testing = {}
kwargs_generator = {'num_workers': 1, 'pin_memory': True} if parameter_dict['USE_GPU'] else {}
use_gpu = torch.cuda.is_available() and  parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")

create_results_dir(parameter_dict['RESULTS_DIR'])
try:
    plot_results(join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                 keys=['loss_registration', 'loss_registration_smoothness', 'loss'])
except:
    print('Plot_results can not be performed. Common causes: no file or uncomplete/repeated rows')


###################################
########### DATA LOADER ###########
###################################
data_loader = parameter_dict['DB_CONFIG']['DATA_LOADER'].DataLoader(parameter_dict['DB_CONFIG'])

num_predictions = np.clip(num_predictions, 0, np.ceil(len(data_loader)))

dataset_test = datasets.IntraModalRegistrationDataset2D(
    data_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    modality='MRI',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7)),
    train=False,
    neighbor_distance=parameter_dict['DISTANCE_NEIGHBOURS']
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
model = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)
model.to(device)

epoch_results_dir = 'model_checkpoint.' + epoch_weights
weightsfile = 'model_checkpoint.' + epoch_weights + '.pth'
checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

da_model = TensorDeformation(image_shape,  parameter_dict['NONLINEAR'].lowres_size, device)

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
        ref_rid = str(rid_image)

        slices_2d = [output_results[9][it_image, 1], output_results[9][it_image, 0]]
        titles = ['Flow_x image', 'Flow_y image']
        slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True, show=False)
        plt.savefig(join(parameter_dict['RESULTS_DIR'], 'results', epoch_results_dir, 'flow_' + ref_rid + '.png'))
        plt.close()

        # ref, flo, reg_r, reg_f = tanh2im([o[it_image] for o in output_results[:4]],
        #                                  mask_list=[o[it_image] for o in output_results[4:8]])
        ref, flo, reg_r, reg_f = [o[it_image] for o in output_results[:4]]
        img_moving = Image.fromarray((255*flo).astype(np.uint8), mode='L')
        img_fixed = Image.fromarray((255*ref).astype(np.uint8), mode='L')
        img_registered_r = Image.fromarray((255*reg_r).astype(np.uint8), mode='L')
        img_registered_f = Image.fromarray((255*reg_f).astype(np.uint8), mode='L')

        frames_dict = { 'initial': [img_fixed, img_moving],
                        'fi': [img_fixed, img_registered_r],
                        'diff': [img_moving, img_registered_r],
                        'fi_rev': [img_moving, img_registered_f],
                        'diff_rev': [img_fixed, img_registered_f]
                    }

        for string, frames in frames_dict.items():
            filepath = join(results_dir, string + '_' + ref_rid +  '.gif')
            frames[0].save(filepath,
                           format='GIF',
                           append_images=frames[1:],
                           save_all=True,
                           duration=1000, loop=0)

print('Predicting done.')
