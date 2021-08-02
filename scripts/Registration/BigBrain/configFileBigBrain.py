import copy
from os.path import join

import numpy as np

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams,CropParams
# from src.utils.image_transforms importMultiplicativeParams, NoiseParams, ClipParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/BigBrain/Registration'

##############################
########### BigBrain  ###########
##############################
CONFIG = {
    'DB_CONFIG': databaseConfig.BigBrain,
    'TRANSFORM':  [CropParams((320, 384), init_coordinates=[0,0])],
    'NORMALIZATION': ScaleNormalization(range=[0,1], quantile=True),
    'DATA_AUGMENTATION': None,
    'ROTATION': AffineParams(rotation=[0]*2, scaling=[0]*2, translation=[0]*2),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,5], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],

    'BATCH_SIZE': 10,
    'N_EPOCHS': 400,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,
    'GPU_INDICES': [0],

    'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [9,9], 'kernel_type': 'mean'}, 'lambda': 1},
    'LOSS_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 0.1},

    'N_CLASSES_SEGMENTATION': False,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'UPSAMPLE_LEVELS': 2,
    'FIELD_TYPE': 'velocity',
    'DISTANCE_NEIGHBOURS': 3,

}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG['LOSS_SMOOTHNESS']['lambda']) + '',
                             'sigmoid_bidir_' + str(CONFIG['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG['RESULTS_DIR'] += '_noDA' if CONFIG['DATA_AUGMENTATION'] is None else '_DA'


CONFIG_HISTO = copy.copy(CONFIG)
CONFIG_HISTO['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG_HISTO['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_HISTO['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_HISTO['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_HISTO['LOSS_SMOOTHNESS']['lambda']) + '',
                             'sigmoid_bidir_' + str(CONFIG_HISTO['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_HISTO['RESULTS_DIR'] += '_noDA' if CONFIG_HISTO['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_INTERMODAL = copy.copy(CONFIG)
# CONFIG_INTERMODAL['LOSS_REGISTRATION'] = {'name': 'NCC', 'params': {'kernel_var': [25,25], 'kernel_type': 'mean'}, 'lambda': 0.1}
CONFIG_INTERMODAL['LOSS_REGISTRATION'] = {'name': 'NMI', 'params': {'bin_centers': np.arange(0, 1, 0.05) + 0.025}, 'lambda': 1}
CONFIG_INTERMODAL['LOSS_SMOOTHNESS']['lambda'] = 1
CONFIG_INTERMODAL['RESULTS_DIR'] = join(BASE_DIR, 'MRI_HISTO_' + CONFIG_INTERMODAL['LOSS_REGISTRATION']['name'],
                                        'D' + str(CONFIG_INTERMODAL['UPSAMPLE_LEVELS']) +
                                        '_R' + str(CONFIG_INTERMODAL['LOSS_REGISTRATION']['lambda']) +
                                        '_S' + str(CONFIG_INTERMODAL['LOSS_SMOOTHNESS']['lambda']) + '',
                                        'sigmoid_bidir_' + str(CONFIG_INTERMODAL['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_INTERMODAL['RESULTS_DIR'] += '_noDA' if CONFIG_INTERMODAL['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_INTERMODAL_WEAKLY = copy.copy(CONFIG_INTERMODAL)
CONFIG_INTERMODAL_WEAKLY['LOSS_REGISTRATION'] = {'name': 'NMI', 'params': {'bin_centers': np.arange(0, 1, 0.05) + 0.025}, 'lambda': 1}
CONFIG_INTERMODAL_WEAKLY['LOSS_LABELS'] = {'name': 'Dice', 'params': {}, 'lambda': 1}
CONFIG_INTERMODAL_WEAKLY['RESULTS_DIR'] = join(BASE_DIR, 'MRI_HISTO_WEAKLY_' + CONFIG_INTERMODAL_WEAKLY['LOSS_REGISTRATION']['name'],
                                               'D' + str(CONFIG_INTERMODAL_WEAKLY['UPSAMPLE_LEVELS']) +
                                               '_R' + str(CONFIG_INTERMODAL_WEAKLY['LOSS_REGISTRATION']['lambda']) +
                                               '_RL' + str(CONFIG_INTERMODAL_WEAKLY['LOSS_LABELS']['lambda']) +
                                               '_S' + str(CONFIG_INTERMODAL_WEAKLY['LOSS_SMOOTHNESS']['lambda']) + '',
                                               'sigmoid_bidir_' + str(CONFIG_INTERMODAL_WEAKLY['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_INTERMODAL_WEAKLY['RESULTS_DIR'] += '_noDA' if CONFIG_INTERMODAL_WEAKLY['DATA_AUGMENTATION'] is None else '_DA'