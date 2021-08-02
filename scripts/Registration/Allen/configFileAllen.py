import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams
# from src.utils.image_transforms importMultiplicativeParams, NoiseParams, ClipParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/Allen_old/Registration'

CONFIG = {
    'DB_CONFIG': databaseConfig.ALLEN_old,
    'TRANSFORM': None,
    'NORMALIZATION': ScaleNormalization(range=[0,1], quantile=True),

    'DATA_AUGMENTATION': None,
    'AFFINE': AffineParams(rotation=[2, 2], scaling=[0.02,0.02], translation=[2, 2]),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,5], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],

    'BATCH_SIZE': 10,
    'N_EPOCHS': 150,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,
    'GPU_INDICES': [0],

    'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [9,9], 'kernel_type': 'mean'}, 'lambda': 1},
    'LOSS_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1},

    'N_CLASSES_SEGMENTATION': False,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'UPSAMPLE_LEVELS': 8,
    'FIELD_TYPE': 'velocity',
    'DISTANCE_NEIGHBOURS': 6,

}


CONFIG['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG['LOSS_SMOOTHNESS']['lambda']) + '',
                             'sigmoid_bidir_' + str(CONFIG['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG['RESULTS_DIR'] += '_noDA' if CONFIG['DATA_AUGMENTATION'] is None else '_DA'


CONFIG_NISSL = copy.copy(CONFIG)
CONFIG_NISSL['LOSS_REGISTRATION'] = {'name': 'NCC', 'params': {'kernel_var': [15, 15], 'kernel_type': 'mean'}, 'lambda': 1}
CONFIG_NISSL['LOSS_SMOOTHNESS'] = {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1}
CONFIG_NISSL['N_EPOCHS'] = 100
CONFIG_NISSL['RESULTS_DIR'] = join(BASE_DIR, 'NISSL_' + CONFIG_NISSL['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_NISSL['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_NISSL['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_NISSL['LOSS_SMOOTHNESS']['lambda']) + '',
                             'sigmoid_bidir_' + str(CONFIG_NISSL['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_NISSL['RESULTS_DIR'] += '_noDA' if CONFIG_NISSL['DATA_AUGMENTATION'] is None else '_DA'


CONFIG_IHC = copy.copy(CONFIG)
CONFIG_IHC['RESULTS_DIR'] = join(BASE_DIR, 'IHC_' + CONFIG_IHC['LOSS_REGISTRATION']['name'],
                                   'D' + str(CONFIG_IHC['UPSAMPLE_LEVELS']) +
                                   '_R' + str(CONFIG_IHC['LOSS_REGISTRATION']['lambda']) +
                                   '_S' + str(CONFIG_IHC['LOSS_SMOOTHNESS']['lambda']) + '',
                                   'sigmoid_bidir_' + str(CONFIG_IHC['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_IHC['RESULTS_DIR'] += '_noDA' if CONFIG_IHC['DATA_AUGMENTATION'] is None else '_DA'
