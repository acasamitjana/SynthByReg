import copy
from os.path import join

from src.utils.image_transforms import ScaleNormalization, NonLinearParams, CropParams, AffineParams
# from src.utils.image_transforms importMultiplicativeParams, NoiseParams, ClipParams
from database import databaseConfig

BASE_DIR = '/home/acasamitjana/Results/Registration/BUNGEE_Tools/P57-16/NetworkTraining/Intramodal'
PROCESSING_SHAPE = (832, 640)

##############################
########### ALLEN  ###########
##############################
CONFIG = {
    'DB_CONFIG': databaseConfig.ERC,
    'TRANSFORM':  [CropParams(crop_shape=PROCESSING_SHAPE)],
    'NORMALIZATION': ScaleNormalization(range=[-1,1], quantile=True),
    # 'DATA_AUGMENTATION': [MultiplicativeParams(value_range=[0.9,1.1], distribution='uniform'),
    #                       NoiseParams(value_range=[-0.1,0.1], distribution='normal'),
    #                       ClipParams(value_range=[-1,1])],
    'DATA_AUGMENTATION': None,
    'ROTATION': AffineParams(rotation=[2]*2, scaling=[0.01]*2, translation=[5]*2),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[1, 6], distribution='uniform'),

    'ENC_NF_REG': [16, 32, 32, 64, 64, 64],
    'DEC_NF_REG': [64, 64, 64, 32, 32, 32, 16],

    'BATCH_SIZE': 10,
    'N_EPOCHS': 200,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,
    'GPU_INDICES': [0],

    # 'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [45,45], 'kernel_type': 'mean'}, 'lambda': 1},
    'LOSS_REGISTRATION': {'name': 'L1', 'params': {}, 'lambda': 1},
    'LOSS_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1},

    'N_CLASSES_SEGMENTATION': False,

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'UPSAMPLE_LEVELS': 4,
    'FIELD_TYPE': 'velocity',
    'DISTANCE_NEIGHBOURS': 4,

}

CONFIG['RESULTS_DIR'] = join(BASE_DIR, 'MRI_' + CONFIG['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG['LOSS_SMOOTHNESS']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG['RESULTS_DIR'] += '_noDA' if CONFIG['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_LFB = copy.copy(CONFIG)
CONFIG_LFB['RESULTS_DIR'] = join(BASE_DIR, 'LFB_' + CONFIG_LFB['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_LFB['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_LFB['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_LFB['LOSS_REGISTRATION']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG_LFB['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_LFB['RESULTS_DIR'] += '_noDA' if CONFIG_LFB['DATA_AUGMENTATION'] is None else '_DA'

CONFIG_HE = copy.copy(CONFIG)
CONFIG_HE['RESULTS_DIR'] = join(BASE_DIR, 'HE_' + CONFIG_HE['LOSS_REGISTRATION']['name'],
                             'D' + str(CONFIG_HE['UPSAMPLE_LEVELS']) +
                             '_R' + str(CONFIG_HE['LOSS_REGISTRATION']['lambda']) +
                             '_S' + str(CONFIG_HE['LOSS_REGISTRATION']['lambda']) + '',
                             'tanh_bidir_' + str(CONFIG_HE['DISTANCE_NEIGHBOURS']) + 'neigh')
CONFIG_HE['RESULTS_DIR'] += '_noDA' if CONFIG_HE['DATA_AUGMENTATION'] is None else '_DA'