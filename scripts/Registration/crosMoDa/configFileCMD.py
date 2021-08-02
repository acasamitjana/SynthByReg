import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, RotationParams, AffineParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/CMD/Registration'
BASE_DIR_SEG = '/home/acasamitjana/Results/RegSyn/CMD/Segmentation'

##############################
########### ALLEN  ###########
##############################
CONFIG_MRI = {
    'DB_CONFIG': databaseConfig.crosMoDa,
    'TRANSFORM': None,
    'NORMALIZATION': ScaleNormalization(range=[-1, 1], quantile=True),
    # 'DATA_AUGMENTATION': [MultiplicativeParams(value_range=[0.9,1.1], distribution='uniform'),
    #                       NoiseParams(value_range=[-0.1,0.1], distribution='normal'),
    #                       ClipParams(value_range=[-1,1])],
    'DATA_AUGMENTATION': None,
    'ROTATION': AffineParams(rotation=[2, 2, 2], scaling=[0.05]*3, translation=[1,1,1]),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9, 9], lowres_strength=[1,3], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64],
    'DEC_NF_REG': [64, 32, 32, 16, 16],

    'BATCH_SIZE': 1,
    'N_EPOCHS': 200,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,
    'GPU_INDICES': [0],

    'LOSS_REGISTRATION': {'name': 'L1', 'params': {}, 'lambda': 1},
    'LOSS_REGISTRATION_LABELS': {'name': 'Dice', 'params': {}, 'lambda': 0.5},
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 0.1},

    'N_CLASSES_SEGMENTATION': 5,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'UPSAMPLE_LEVELS': 2,
    'FIELD_TYPE': 'velocity',
    'DISTANCE_NEIGHBOURS': 0,

}

CONFIG_MRI['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG_MRI['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_MRI['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_MRI['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_MRI['LOSS_REGISTRATION_SMOOTHNESS']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG_MRI['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_MRI['RESULTS_DIR'] += '_noDA' if CONFIG_MRI['DATA_AUGMENTATION'] is None else '_DA'
