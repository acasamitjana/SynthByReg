import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/L2R/InfoNCE'

CONFIG = {
    'DB_CONFIG': databaseConfig.L2R,
    'TRANSFORM': None,
    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0,1], quantile=False),

    'ROTATION': AffineParams(rotation=[0]*3, scaling=[0]*3, translation=[0]*3),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9, 9], lowres_strength=[0,0], distribution='uniform'),

    # 'ENC_NF_REG': [8, 16, 16, 32],
    # 'DEC_NF_REG': [32, 16, 16, 16, 16, 8, 8],
    # 'ENC_NF_REG': [8, 16, 32, 64],
    # 'DEC_NF_REG': [64, 32, 16, 16, 8],
    'ENC_NF_REG': [16, 32, 32, 64],
    'DEC_NF_REG': [64, 32, 32, 16, 16],

    'NUM_IMAGES': None,
    'BATCH_SIZE': 1,
    'N_EPOCHS': 300,
    'LEARNING_RATE': {'generator': 2e-4, 'discriminator': 2e-4}, #{'generator': 2e-5, 'discriminator': 2e-5},
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'WEIGHTS_SEGMENTATION_MRI_TANH': '/home/acasamitjana/Results/RegSyn/L2R/Segmentation/MRI_Dice/D2_S0.5/tanh_bidir_0neigh_noDA/checkpoints/model_checkpoint.FI.pth',
    'WEIGHTS_REGISTRATION_MRI_TANH': '/home/acasamitjana/Results/RegSyn/L2R/Registration/MRI_L1/D2_R1_S0.1/tanh_bidir_0neigh_noDA_weakly/checkpoints/model_checkpoint.FI.pth',
    'WEIGHTS_REGISTRATION_MRI_SIGMOID': '/home/acasamitjana/Results/RegSyn/L2R/Registration/MRI_L1/D1_R1_S1/sigmoid_bidir_0neigh_noDA_weakly/checkpoints/model_checkpoint.FI.pth',
    'WEIGHTS_REGISTRATION_HISTO_TANH': '',

    'N_CLASSES_SEGMENTATION': 5,
    'N_CLASSES_NETWORK': 8,
    'UPSAMPLE_LEVELS': 1,

    'LOSS_REGISTRATION_NCC': 'NCC',
    'LOSS_REGISTRATION_L1': 'L1',
    'LOSS_REGISTRATION_SMOOTHNESS': 'Grad',
    'LOSS_GAN': 'LSGAN',

    'LAMBDA_GAN': 1,
    'LAMBDA_REGISTRATION_NCC': 0,
    'LAMBDA_REGISTRATION_LABELS': 0,
    'LAMBDA_REGISTRATION_L1': 5,
    'LAMBDA_REGISTRATION_SMOOTHNESS': 0.1,
    'LAMBDA_NCE': 0.5,

    'TEMPERATURE': 0.05,
    'NUM_LAYERS_DISCRIMINATOR': 3,
    'FIELD_TYPE': 'displacement',

    'USE_GPU': True,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'PARENT_DIRECTORY': '2_NCE_SIGMOID'
}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, CONFIG['PARENT_DIRECTORY'], 'INITIAL')