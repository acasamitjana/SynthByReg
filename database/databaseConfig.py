from os.path import join

from setup import DATA_DIR
from database.Allen import data_loader_subset as DL_Allen_subset
from database.BigBrain import data_loader as DL_BigBrain


ALLEN_subset = {
    'BASE_DIR': join(DATA_DIR, 'Allen_subset_labels'),
    'HISTOLOGY_DIR': join(DATA_DIR, 'Allen_subset_labels','histo'),
    'MRI_DIR': join(DATA_DIR, 'Allen_subset_labels','mri'),
    'LANDMARKS_DIR': join(DATA_DIR, 'Allen_subset_labels', 'landmarks'),
    'NAME': 'Allen_subset',
    'DATA_FILE': 'slice_id.txt',
    'DATA_LOADER': DL_Allen_subset
}

BigBrain = {
    'BASE_DIR': join(DATA_DIR, 'BigBrain'),
    'MRI_DIR': join(DATA_DIR, 'BigBrain', 'mri'),
    'HISTOLOGY_DIR': join(DATA_DIR, 'BigBrain', 'histo'),
    'LANDMARKS_DIR': join(DATA_DIR, 'BigBrain', 'landmarks'),
    'NAME': 'BigBrain',
    'DATA_FILE': 'slice_id.txt',
    'DATA_LOADER': DL_BigBrain
}
