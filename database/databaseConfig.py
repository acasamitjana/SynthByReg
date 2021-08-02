from os.path import join

from database.Allen import data_loader as DL_Allen, data_loader_subset as DL_Allen_subset
from database.L2R_CTMR import data_loader as DL_L2R
from database.ERC import data_loader as DL_ERC
from database.Cricks import data_loader as DL_Cricks
from database.BigBrain import data_loader as DL_BigBrain
# It is important that all files contain the same structure: prefix + '_' + slice_num + '.' + extension

SLICE_PREFIX = 'slice'
BASE_DIR = '/home/acasamitjana/Data'

# With Allen I have converted images to mode='L' and masked using the label masks.
ALLEN_old = {
    'NAME': 'ALLEN',
    'BASE_DIR': join(BASE_DIR, 'Allen_paper'),
    'NISSL_DIR': join(BASE_DIR, 'Allen_paper',  'nissl'),
    'IHC_DIR': join(BASE_DIR, 'Allen_paper', 'ihc'),
    'MRI_DIR': join(BASE_DIR, 'Allen_paper', 'mri'),
    'SLICE_PREFIX': 'slice',
    'IMAGE_EXTENSION': '.png',
    'NISSL_FILE': join(BASE_DIR, 'Allen_paper', 'nissl', 'slice_separation_extended.csv'),
    'IHC_FILE': join(BASE_DIR, 'Allen_paper',  'ihc', 'slice_separation_extended.csv'),
    'DATA_LOADER': DL_Allen
}

ALLEN = {
    'NAME': 'ALLEN',
    'BASE_DIR': join(BASE_DIR, 'Allen', 'dataset'),
    'NISSL_DIR': join(BASE_DIR, 'Allen', 'dataset', 'nissl'),
    'IHC_DIR': join(BASE_DIR, 'Allen', 'dataset', 'ihc'),
    'MRI_DIR': join(BASE_DIR, 'Allen', 'dataset','mri'),
    'SLICE_PREFIX': 'slice',
    'IMAGE_EXTENSION': '.png',
    'NISSL_FILE': join(BASE_DIR, 'Allen', 'dataset', 'nissl', 'slice_separation.csv'),
    'IHC_FILE': join(BASE_DIR, 'Allen', 'dataset', 'ihc', 'slice_separation.csv'),
    'DATA_LOADER': DL_Allen
}

ALLEN_subset = {
    'BASE_DIR': join(BASE_DIR, 'Allen_subset_labels'),
    'HISTOLOGY_DIR': join(BASE_DIR, 'Allen_subset_labels','histo'),
    'MRI_DIR': join(BASE_DIR, 'Allen_subset_labels','mri'),
    'LANDMARKS_DIR': join(BASE_DIR, 'Allen_subset_labels', 'landmarks'),
    'SLICE_PREFIX': 'slice',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'ALLEN',
    'DATA_FILE': 'slice_id.txt',
    'DATA_LOADER': DL_Allen_subset
}

ALLEN_subset_cll = {
    'BASE_DIR': join(BASE_DIR, 'Allen_subset_labels'),
    'HISTOLOGY_DIR': join(BASE_DIR, 'Allen_subset_labels','histo'),
    'MRI_DIR': join(BASE_DIR, 'Allen_subset_labels','mri'),
    'LANDMARKS_DIR': join(BASE_DIR, 'Allen_subset_labels', 'landmarks'),
    'SLICE_PREFIX': 'slice',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'ALLEN',
    'DATA_FILE': 'slice_id_cerebellum.txt',
    'DATA_LOADER': DL_Allen_subset
}

ALLEN_subset_cr = {
    'BASE_DIR': join(BASE_DIR, 'Allen_subset_labels'),
    'HISTOLOGY_DIR': join(BASE_DIR, 'Allen_subset_labels','histo'),
    'MRI_DIR': join(BASE_DIR, 'Allen_subset_labels','mri'),
    'LANDMARKS_DIR': join(BASE_DIR, 'Allen_subset_labels', 'landmarks'),
    'SLICE_PREFIX': 'slice',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'ALLEN',
    'DATA_FILE': 'slice_id_cerebrum.txt',
    'DATA_LOADER': DL_Allen_subset
}

L2R = {
    'DATA_DIR': '/home/acasamitjana/Data/Learn2Reg/Task01/Train',
    'DATA_FILE': '/home/acasamitjana/Data/Learn2Reg/Task01/Train/subject_list.csv',
    'DATA_LOADER': DL_L2R
}

crosMoDa = {
    'DATA_DIR': '/home/acasamitjana/Data/crossmoda-training/Task01/Train',
    'DATA_FILE': '/home/acasamitjana/Data/Learn2Reg/Task01/Train/subject_list.csv',
    'DATA_LOADER': DL_L2R
}

ERC = {
    'BASE_DIR': '/home/acasamitjana/Results/Registration/RigidRegistration/P57-16/Histo_GAP_LBFGS_v2/results/slices/',
    'SLICE_PREFIX': 'slice_',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'BUNGEE_Tools',
    'DATA_LOADER': DL_ERC
}

Cricks = {
    'BASE_DIR': join(BASE_DIR, 'Cricks'),
    'NEUN_DIR': join(BASE_DIR, 'Cricks', 'neun'),
    'DAPI_DIR': join(BASE_DIR, 'Cricks', 'dapi'),
    'MRI_DIR': join(BASE_DIR, 'Cricks', 'mri'),
    'SLICE_PREFIX': 'slice',
    'IMAGE_EXTENSION': '.png',
    'DATA_FILE': join(BASE_DIR, 'Cricks', 'slice_separation.csv'),
    'NAME': 'Cricks',
    'DATA_LOADER': DL_Cricks
}

BigBrain = {
    'BASE_DIR': join(BASE_DIR, 'BigBrain'),
    'MRI_DIR': join(BASE_DIR, 'BigBrain', 'mri'),
    'HISTOLOGY_DIR': join(BASE_DIR, 'BigBrain', 'histo'),
    'LANDMARKS_DIR': join(BASE_DIR, 'BigBrain', 'landmarks'),
    'NAME': 'BigBrain',
    'DATA_LOADER': DL_BigBrain
}