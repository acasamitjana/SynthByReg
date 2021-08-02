import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams, CropParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/Registration/BUNGEE_Tools/P57-16/NetworkTraining/Intermodal'
PROCESSING_SHAPE = (832, 640)

CONFIG = {
    'DB_CONFIG': databaseConfig.ERC,
    'TRANSFORM':  [CropParams(crop_shape=PROCESSING_SHAPE)],
    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[-1,1], quantile=True),

    'ROTATION': AffineParams(rotation=[0,0], scaling=[0,0], translation=[0,0]),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,5], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],


    'NUM_IMAGES': None,
    'BATCH_SIZE': 1,
    'N_EPOCHS': 400,
    'LEARNING_RATE': {'generator': 2e-4, 'discriminator': 2e-4}, #{'generator': 2e-5, 'discriminator': 2e-5},
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'WEIGHTS_SEGMENTATION': '',
    'WEIGHTS_REGISTRATION_MRI_TANH': '/home/acasamitjana/Results/Registration/BUNGEE_Tools/P57-16/NetworkTraining/Intramodal/MRI_L1/D4_R1_S1/tanh_bidir_4neigh_noDA',
    'WEIGHTS_REGISTRATION_HISTO_TANH': '',

    'N_CLASSES_SEGMENTATION': 4,
    'N_CLASSES_NETWORK': 8,
    'UPSAMPLE_LEVELS': 4,

    'LOSS_NCE': {'name': 'NCE', 'params': {'nce_T': 0.01, 'batch_size': 1}, 'lambda': 0.1},
    'LOSS_REGISTRATION_NCC': {'name': 'NCC', 'params': {'kernel_var': [9, 9], 'kernel_type': 'mean'}, 'lambda': 0},
    'LOSS_REGISTRATION_L1': {'name': 'L1', 'params': {}, 'lambda': 1},
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1},
    'LOSS_GAN':  {'name': 'LSGAN', 'params': {}, 'lambda': 1},

    'NUM_LAYERS_DISCRIMINATOR': 3,
    'FIELD_TYPE': 'velocity',

    'USE_GPU': True,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'PARENT_DIRECTORY': '1_NCE_TANH'
}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, CONFIG['PARENT_DIRECTORY'])