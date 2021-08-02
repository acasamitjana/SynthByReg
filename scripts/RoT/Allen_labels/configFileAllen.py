import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams, CropParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/Allen_labels/RoT/'

CONFIG = {
    'DB_CONFIG': databaseConfig.ALLEN_subset,
    'TRANSFORM': [CropParams((448, 320), init_coordinates=[0, 0])],

    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0, 1], quantile=False),

    'ROTATION': AffineParams(rotation=[0,0], scaling=[0,0], translation=[0,0]),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,5], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],


    'NUM_IMAGES': None,
    'BATCH_SIZE': 1,
    'N_EPOCHS': 400,
    'LEARNING_RATE': {'generator': 2e-4, 'discriminator': 2e-4},
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'WEIGHTS_SEGMENTATION': '',
    'WEIGHTS_REGISTRATION_MRI_TANH': '',
    'WEIGHTS_REGISTRATION_MRI_SIGMOID': '/home/acasamitjana/Results/RegSyn/Allen_labels/Registration/MRI_NCC/D2_R1_S0.1/sigmoid_bidir_2neigh_noDA/checkpoints/model_checkpoint.BO.pth',

    'N_CLASSES_SEGMENTATION': 5,
    'N_CLASSES_NETWORK': 8,
    'UPSAMPLE_LEVELS': 2,

    'LOSS_REGISTRATION_NCC': 'NCC',
    'LOSS_REGISTRATION_L1': 'L1',
    'LOSS_REGISTRATION_SMOOTHNESS': 'Grad',
    'LOSS_GAN': 'LSGAN',

    'LAMBDA_GAN': 1,
    'LAMBDA_REGISTRATION_NCC': 1,
    'LAMBDA_REGISTRATION_L1': 1,
    'LAMBDA_REGISTRATION_SMOOTHNESS': 1,
    'LAMBDA_NCE': 0.5,

    'TEMPERATURE': 0.01,
    'NUM_LAYERS_DISCRIMINATOR': 3,
    'FIELD_TYPE': 'velocity',

    'USE_GPU': True,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'PARENT_DIRECTORY': ''
}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, CONFIG['PARENT_DIRECTORY'], '')
