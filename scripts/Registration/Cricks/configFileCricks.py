import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, AffineParams, CropParams
# from src.utils.image_transforms importMultiplicativeParams, NoiseParams, ClipParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/Cricks/Registration'

##############################
########### ALLEN  ###########
##############################
CONFIG = {
    'DB_CONFIG': databaseConfig.Cricks,
    'TRANSFORM': [CropParams((576,768), init_coordinates=[0,0])],
    'NORMALIZATION': ScaleNormalization(range=[0,1], quantile=True),

    'DATA_AUGMENTATION': None,
    'ROTATION': AffineParams(rotation=[2]*2, scaling=[0.01]*2, translation=[1]*2),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,9], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],

    'BATCH_SIZE': 10,
    'N_EPOCHS': 300,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,
    'GPU_INDICES': [0],

    'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [15,15], 'kernel_type': 'mean'}, 'lambda': 1},#
    'LOSS_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1},

    'N_CLASSES_SEGMENTATION': False,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'UPSAMPLE_LEVELS': 2,
    'FIELD_TYPE': 'velocity',
    'DISTANCE_NEIGHBOURS': 0,

}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG['LOSS_SMOOTHNESS']['lambda']) + '',
                             'sigmoid_bidir_' + str(CONFIG['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG['RESULTS_DIR'] += '_noDA' if CONFIG['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_DAPI = copy.copy(CONFIG)
CONFIG_DAPI['LOSS_REGISTRATION'] = {'name': 'L1', 'params': {}, 'lambda': 1}
CONFIG_DAPI['UPSAMPLE_LEVELS'] = 2
CONFIG_DAPI['LOSS_SMOOTHNESS']['lambda'] = 0.5
CONFIG_DAPI['RESULTS_DIR'] = join(BASE_DIR, 'DAPI_' + CONFIG_DAPI['LOSS_REGISTRATION']['name'],
                                 'D' + str(CONFIG_DAPI['UPSAMPLE_LEVELS']) +
                                 '_R' + str(CONFIG_DAPI['LOSS_REGISTRATION']['lambda']) +
                                 '_S' + str(CONFIG_DAPI['LOSS_SMOOTHNESS']['lambda']) + '',
                                 'tanh_bidir_' + str(CONFIG_DAPI['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_DAPI['RESULTS_DIR'] += '_noDA' if CONFIG_DAPI['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_NEUN = copy.copy(CONFIG)
CONFIG_NEUN['RESULTS_DIR'] = join(BASE_DIR, 'DAPI_' + CONFIG_NEUN['LOSS_REGISTRATION']['name'],
                                 'D' + str(CONFIG_NEUN['UPSAMPLE_LEVELS']) +
                                 '_R' + str(CONFIG_NEUN['LOSS_REGISTRATION']['lambda']) +
                                 '_S' + str(CONFIG_NEUN['LOSS_SMOOTHNESS']['lambda']) + '',
                                 'tanh_bidir_' + str(CONFIG_NEUN['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_NEUN['RESULTS_DIR'] += '_noDA' if CONFIG_NEUN['DATA_AUGMENTATION'] is None else '_DA'