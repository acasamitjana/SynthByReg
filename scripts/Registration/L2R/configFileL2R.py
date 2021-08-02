import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, RotationParams, AffineParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/RegSyn/L2R/Registration'
BASE_DIR_SEG = '/home/acasamitjana/Results/RegSyn/L2R/Segmentation'

##############################
########### ALLEN  ###########
##############################
CONFIG_MRI = {
    'DB_CONFIG': databaseConfig.L2R,
    'TRANSFORM': None,
    'NORMALIZATION': ScaleNormalization(range=[0, 1], quantile=True),
    'DATA_AUGMENTATION': None,
    'ROTATION': AffineParams(rotation=[5]*3, scaling=[0.05]*3, translation=[3]*3),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9, 9], lowres_strength=[2,5], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64],
    'DEC_NF_REG': [64, 32, 32, 16, 16],

    'BATCH_SIZE': 1,
    'N_EPOCHS': 300,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,
    'GPU_INDICES': [0],

    'LOSS_REGISTRATION': {'name': 'L1', 'params': {}, 'lambda': 1}, #{'name': 'NCC', 'params': {'kernel_var': [15,15,15], 'kernel_type': 'mean'}, 'lambda': 1},
    'LOSS_REGISTRATION_LABELS': {'name': 'Dice', 'params': {}, 'lambda': 1},
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 3, 'penalty': 'l2'}, 'lambda': 1},

    'N_CLASSES_SEGMENTATION': 5,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'UPSAMPLE_LEVELS': 1,
    'FIELD_TYPE': 'deformation',
    'DISTANCE_NEIGHBOURS': 0,

}

CONFIG_MRI['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG_MRI['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_MRI['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_MRI['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_MRI['LOSS_REGISTRATION_SMOOTHNESS']['lambda']) + '',
                             'sigmoid_bidir_' + str(CONFIG_MRI['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_MRI['RESULTS_DIR'] += '_noDA' if CONFIG_MRI['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_CT = copy.copy(CONFIG_MRI)
CONFIG_CT['RESULTS_DIR'] = join(BASE_DIR, 'CT_' + CONFIG_CT['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_CT['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_CT['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_CT['LOSS_REGISTRATION_SMOOTHNESS']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG_CT['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_CT['RESULTS_DIR'] += '_noDA' if CONFIG_CT['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_MRI_SEG = copy.copy(CONFIG_MRI)
CONFIG_MRI_SEG['LOSS_SEGMENTATION'] = {'name': 'Dice', 'params': {}, 'lambda': 1}
# CONFIG_MRI_SEG['ROTATION'] = AffineParams(rotation=[10]*3, scaling=[0.1]*3, translation=[5]*3)
CONFIG_MRI_SEG['RESULTS_DIR'] = join(BASE_DIR_SEG, 'MRI_' + CONFIG_MRI_SEG['LOSS_SEGMENTATION']['name'],
                             'D' + str(CONFIG_MRI_SEG['UPSAMPLE_LEVELS']) +
                             '_S' + str(CONFIG_MRI_SEG['LOSS_SEGMENTATION']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG_MRI_SEG['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_MRI_SEG['RESULTS_DIR'] += '_noDA' if CONFIG_MRI_SEG['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_CT_SEG = copy.copy(CONFIG_MRI_SEG)
CONFIG_CT_SEG['RESULTS_DIR'] = join(BASE_DIR_SEG, 'CT_' + CONFIG_CT_SEG['LOSS_SEGMENTATION']['name'],
                             'D' + str(CONFIG_CT_SEG['UPSAMPLE_LEVELS']) +
                             '_S' + str(CONFIG_CT_SEG['LOSS_SEGMENTATION']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG_CT_SEG['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_CT_SEG['RESULTS_DIR'] += '_noDA' if CONFIG_CT_SEG['DATA_AUGMENTATION'] is None else '_DA'