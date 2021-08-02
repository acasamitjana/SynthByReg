import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/Allen/InfoNCE/'

CONFIG = {
    'DB_CONFIG': databaseConfig.ALLEN,
    'TRANSFORM': None,
    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0,1], quantile=True),

    'ROTATION': AffineParams(rotation=[2, 2], scaling=[0.02,0.02], translation=[2, 2]),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,5], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],


    'NUM_IMAGES': None,
    'BATCH_SIZE': 1,
    'N_EPOCHS': 150,
    'LEARNING_RATE': {'generator': 2e-4, 'discriminator': 2e-4}, #{'generator': 2e-5, 'discriminator': 2e-5},
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'WEIGHTS_SEGMENTATION': '',
    'WEIGHTS_REGISTRATION_MRI_SIGMOID': '/home/acasamitjana/Results/RegSyn/Allen/Registration/MRI_NCC/D2_R1_S1/sigmoid_bidir_6neigh_noDA/checkpoints/model_checkpoint.FI.pth',#'/home/acasamitjana/Results/RegSyn/Allen/Registration/MRI_L1/D4_R1_S1/tanh_bidir_3neigh_noDA/checkpoints/model_checkpoint.BO.pth',
    # 'WEIGHTS_REGISTRATION_MRI_SIGMOID': '/home/acasamitjana/Results/RegSyn/Allen_old/Registration/MRI_NCC/D8_R1_S1/sigmoid_bidir_6neigh_noDA/checkpoints/model_checkpoint.FI.pth',#'/home/acasamitjana/Results/RegSyn/Allen/Registration/MRI_L1/D4_R1_S1/tanh_bidir_3neigh_noDA/checkpoints/model_checkpoint.BO.pth',
    'WEIGHTS_REGISTRATION_HISTO_SIGMOID': '/home/acasamitjana/Results/RegSyn/Allen/Registration/NISSL_NCC/D2_R1_S1/sigmoid_bidir_6neigh_noDA/checkpoints/model_checkpoint.FI.pth',
    # 'WEIGHTS_REGISTRATION_HISTO_SIGMOID': '/home/acasamitjana/Results/RegSyn/Allen_old/Registration/NISSL_NCC/D8_R1_S1/sigmoid_bidir_6neigh_noDA/checkpoints/model_checkpoint.FI.pth',

    'N_CLASSES_SEGMENTATION': 4,
    'N_CLASSES_NETWORK': 8,
    'UPSAMPLE_LEVELS': 8,

    'LOSS_REGISTRATION_NCC': 'NCC',
    'LOSS_REGISTRATION_L1': 'L1',
    'LOSS_REGISTRATION_SMOOTHNESS': 'Grad',
    'LOSS_GAN': 'LSGAN',

    'LAMBDA_GAN': 1,
    'LAMBDA_REGISTRATION_NCC': 0,
    'LAMBDA_REGISTRATION_L1': 1,
    'LAMBDA_REGISTRATION_SMOOTHNESS': 1,
    'LAMBDA_NCE': 0.02,

    'TEMPERATURE': 0.05,
    'NUM_LAYERS_DISCRIMINATOR': 3,
    'FIELD_TYPE': 'velocity',

    'USE_GPU': True,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'PARENT_DIRECTORY': '2_NCE_SIGMOID'
}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, CONFIG['PARENT_DIRECTORY'], '')

CONFIG_BO = copy.copy(CONFIG)
CONFIG_BO['PARENT_DIRECTORY'] += '/BO'
CONFIG_BO['RESULTS_DIR'] = join(BASE_DIR, CONFIG_BO['PARENT_DIRECTORY'], '')