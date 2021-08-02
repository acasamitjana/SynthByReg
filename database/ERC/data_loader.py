import csv
from os.path import join, exists
import warnings

import numpy as np
from PIL import Image

class Slice(object):

    def __init__(self, sid, data_path, stain_flag, slice_prefix=None, block_structures=None, id_mm=None):

        self._id = sid
        self._id_mm = id_mm if id_mm is not None else int(sid)

        self.data_path = data_path
        self.block_structures = [int(l) for l in block_structures.split(' ')]

        if slice_prefix is None:
            slice_prefix = ''

        self.image_MRI = join(data_path, 'MRI', slice_prefix + str(sid) + '.png')
        self.image_HISTO = join(data_path, stain_flag + '_gray_affine' ,slice_prefix + str(sid) + '.png')

        self.mask_MRI = join(data_path, 'MRI_masks', slice_prefix + str(sid) + '.png')
        self.mask_HISTO = join(data_path, stain_flag + '_masks_affine' ,slice_prefix + str(sid) + '.png')

        self.affine_HISTO = join(data_path, stain_flag + '_affine', slice_prefix + str(sid) + '.aff')

        self.labels_MRI = join(data_path, 'MRI_labels', slice_prefix + str(sid) + '.png')
        self.labels_HISTO = join(data_path, stain_flag + '_labels_affine', slice_prefix + str(sid) + '.png')

    @property
    def id(self):
        return self._id

    @property
    def id_mm(self):
        return float(self._id_mm)

    @property
    def image_shape(self):
        mask = self.load_mri()
        return mask.shape

    def load_mri(self, *args, **kwargs):
        data = Image.open(self.image_MRI)
        data = np.array(data)

        return data

    def load_histo(self, *args, **kwargs):
        return self.load_histo_affine

    def load_histo_affine(self, *args, **kwargs):
        data = Image.open(self.image_HISTO)
        data = np.array(data)

        return data

    def load_mri_mask(self, *args, **kwargs):
        data = Image.open(self.mask_MRI)
        data = np.array(data)

        return data

    def load_histo_mask(self, *args, **kwargs):
        return self.load_histo_mask_affine

    def load_histo_mask_affine(self, *args, **kwargs):
        data = Image.open(self.mask_HISTO)
        data = np.array(data)

        return data

    def load_mri_labels(self, *args, **kwargs):
        return self.load_mri_mask()

    def load_histo_labels(self, *args, **kwargs):
        return self.load_histo_mask()

    def load_histo_labels_affine(self, *args, **kwargs):
        return self.load_histo_mask_affine()

    def load_mri_landmarks(self, *args, **kwargs):
        return []

    def load_mri_landmarks_affine(self, *args, **kwargs):
        return []

    def load_histo_landmarks(self, *args, **kwargs):
        return []

    def load_histo_landmarks_affine(self, *args, **kwargs):
        return []

    def load_affine_histo2mri(self, full=False, *args, **kwargs):
        affine_matrix = np.zeros((2,3))

        with open(self.affine_HISTO, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ' ')
            row = next(csvreader)
            affine_matrix[0, 0] = float(row[0])
            affine_matrix[0, 1] = float(row[1])
            affine_matrix[0, 2] = float(row[3])
            row = next(csvreader)
            affine_matrix[1, 0] = float(row[0])
            affine_matrix[1, 1] = float(row[1])
            affine_matrix[1, 2] = float(row[3])


        return affine_matrix


class Block(object):
    def __init__(self, bid, data_path, stain_flag, structures=None):

        self.n_dims = 2
        self.n_channels = 1

        self.data_path = data_path
        self._bid = bid
        self.stain_flag = stain_flag

        self.image_MRI = join(data_path, 'MRI')
        self.image_LFB = join(data_path, 'LFB_gray_affine')
        self.image_HE = join(data_path, 'HE_gray_affine')

        self.structures = [int(l) for l in structures.split(' ')] if structures is not None else None

        self.subject_dict = {}

    def get_slice(self, random_seed=44, slice_id=None):

        if slice_id is None:
            np.random.seed(random_seed)
            slice_id = np.random.choice(self.rid_list, 1, replace=False)[0]

        s = self.subject_dict[slice_id]

        return s, slice_id

    def add_slice(self, slice):
        self.subject_dict[slice.id] = slice

    def __len__(self):
        return len(self.subject_dict.keys())

    @property
    def subject_list(self):
        return [sbj for sbj in self.subject_dict.values()]

    @property
    def rid_list(self):
        return [sbj for sbj in self.subject_dict.keys()]

    @property
    def id(self):
        return self._bid

    @property
    def vol_shape(self):
        return self.image_shape + (len(self.subject_dict.keys()),)

    @property
    def num_classes(self):
        if self.structures is None:
            return 0
        else:
            return len(self.structures)

    @property
    def image_shape(self):
        return list(self.subject_dict.values())[0].image_shape


class DataLoader(object):

    def __init__(self, database_config, **kwargs):
        self.database_config = database_config
        self.data_path = database_config['BASE_DIR']
        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, bid_list=None, stain_flag='LFB'):
        self.n_dims = 2
        self.n_channels = 1

        self.subject_dict = {}
        with open(join(self.database_config['BASE_DIR'], 'slice_id.txt'), 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if bid_list is not None:
                    if row['BLOCK_ID'] not in bid_list:
                        continue

                if row['BLOCK_ID'] not in self.subject_dict.keys():
                    self.subject_dict[row['BLOCK_ID']] = Block(row['BLOCK_ID'],
                                                               self.database_config['BASE_DIR'],
                                                               stain_flag,
                                                               row['STRUCTURES'])

                self.subject_dict[row['BLOCK_ID']].add_slice(
                    Slice(sid=row['SLICE_ID'], data_path=join(self.database_config['BASE_DIR'], row['BLOCK_ID']),
                          stain_flag=stain_flag, slice_prefix=self.database_config['SLICE_PREFIX'], block_structures=row['STRUCTURES'])
                )


        self.rid_list = [sbj for sbj in self.subject_dict.keys()]
        self.subject_list = [sbj for sbj in self.subject_dict.values()]

        self.sid_list = [s.id for sbj in self.subject_dict.values() for s in sbj.subject_list]
        self.slice_list = [s for sbj in self.subject_dict.values() for s in sbj.subject_list]

    @property
    def image_shape(self):
        return self.subject_list[0].image_shape

    def __len__(self):
        return sum([len(block) for block in self.subject_dict.values()])

