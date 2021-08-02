import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams, CropParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/BigBrain/CycleGAN'

CONFIG = {
    'DB_CONFIG': databaseConfig.BigBrain,
    'TRANSFORM':  [CropParams((320, 384), init_coordinates=[0,0])],
    'NORMALIZATION': ScaleNormalization(range=[0,1], quantile=True),
    'DATA_AUGMENTATION': None,
    'ROTATION': AffineParams(rotation=[0]*2, scaling=[0]*2, translation=[0]*2),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[1,3], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],

    'NUM_IMAGES': None,
    'BATCH_SIZE': 1,
    'N_EPOCHS': 400,
    'LEARNING_RATE': {'generator': 2e-4, 'discriminator': 2e-4}, #{'generator': 2e-5, 'discriminator': 2e-5},
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'WEIGHTS_SEGMENTATION': '',
    'WEIGHTS_REGISTRATION_MRI_TANH': '',
    'WEIGHTS_REGISTRATION_MRI_SIGMOID': '/home/acasamitjana/Results/RegSyn/BigBrain/Registration/MRI_NCC/D2_R1.0_S0.1/sigmoid_bidir_3neigh_noDA/checkpoints/model_checkpoint.BO.pth',
    'WEIGHTS_REGISTRATION_HISTO_TANH': '',

    'N_CLASSES_SEGMENTATION': 4,
    'N_CLASSES_NETWORK': 8,
    'UPSAMPLE_LEVELS': 2,

    'LOSS_REGISTRATION': 'L1',
    'LOSS_REGISTRATION_SMOOTHNESS': 'Grad',
    'LOSS_GAN': 'LSGAN',
    'LOSS_CYCLE': 'L1',

    'LAMBDA_GAN': 1,
    'LAMBDA_REGISTRATION': 1,
    'LAMBDA_REGISTRATION_SMOOTHNESS': 1,
    'LAMBDA_CYCLE': 1,

    'NUM_LAYERS_DISCRIMINATOR': 3,
    'FIELD_TYPE': 'velocity',

    'USE_GPU': True,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'PARENT_DIRECTORY': '2_NCE_SIGMOID'
}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, CONFIG['PARENT_DIRECTORY'], '')