import csv
from os.path import join
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_opening


class Subject():
    def __init__(self, id, data_path, mri_data, mri_mask, histo_data, histo_mask, id_mm=None):

        self.data_path = data_path
        self._id = id
        self._id_mm = id_mm if id_mm is not None else id

        self.image_mri = mri_data
        self.image_histo = histo_data
        self.image_histo_affine = histo_data

        self.labels_mri = mri_mask
        self.labels_histo = histo_mask
        self.labels_histo_affine = histo_mask

        self.landmarks = None
        self.landmarks_affine = None


    @property
    def id(self):
        return self._id

    @property
    def id_mm(self):
        return int(self._id_mm)

    @property
    def image_shape(self):
        mask = self.load_mri()
        return mask.shape

    def load_mri(self, *args, **kwargs):
        return self.image_mri[..., int(self._id)]

    def load_histo(self,*args, **kwargs):
        return self.image_histo[..., int(self._id)]

    def load_histo_affine(self, *args, **kwargs):
        return self.image_histo_affine[..., int(self._id)]

    def load_mri_mask(self, *args, **kwargs):
        data = self.labels_mri[..., int(self._id)]/np.max(self.labels_mri[..., int(self._id)]) > 0.5
        # data = binary_opening(data, structure=np.ones((5,5)))
        return data > 0.5

    def load_histo_mask(self, *args, **kwargs):
        data = self.labels_histo[..., int(self._id)]/np.max(self.labels_histo[..., int(self._id)]) > 0.5
        return data / np.max(data) > 0.5

    def load_histo_mask_affine(self, *args, **kwargs):
        data = self.labels_histo_affine[..., int(self._id)]/np.max(self.labels_histo_affine[..., int(self._id)]) > 0.5
        return data / np.max(data) > 0.5

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
        return np.eye(4)


class DataLoader(object):

    def __init__(self, database_config, **kwargs):
        self.database_config = database_config
        self.data_path = database_config['BASE_DIR']
        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, stain='NEUN', rid_list=None):
        self.n_dims = 2
        self.n_channels = 1

        self.subject_list = []
        self.rid_list = []
        self.subject_dict = {}

        if stain == 'Combined':
            self.database_config['MRI_DIR'] = self.database_config['NEUN_DIR']
            stain = 'DAPI'

            proxy = nib.load(join(self.database_config['NEUN_DIR'], 'NEUN_images_0.02mm_affine.nii.gz'))
            mri_data = proxy.get_fdata()

            proxy = nib.load(join(self.database_config['NEUN_DIR'], 'NEUN_masks_0.02mm_affine.nii.gz'))
            mri_mask = proxy.get_fdata()

        else:
            proxy = nib.load(join(self.database_config['MRI_DIR'], 'MRI_images_0.02mm.nii.gz'))
            mri_data = proxy.get_fdata()

            proxy = nib.load(join(self.database_config['MRI_DIR'], 'MRI_masks_0.02mm.nii.gz'))
            mri_mask = proxy.get_fdata()

        proxy = nib.load(join(self.database_config[stain + '_DIR'], stain + '_images_0.02mm_affine.nii.gz'))
        histo_data = proxy.get_fdata()

        proxy = nib.load(join(self.database_config[stain + '_DIR'], stain + '_masks_0.02mm_affine.nii.gz'))
        histo_mask = proxy.get_fdata()
        with open(self.database_config['DATA_FILE'], 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if rid_list is not None:
                    if row['slice_num'] not in rid_list:
                        continue

                subject = Subject(id=row['slice_num'],
                                  data_path=self.database_config['BASE_DIR'],
                                  mri_data=mri_data,
                                  mri_mask=mri_mask,
                                  histo_data=histo_data,
                                  histo_mask=histo_mask
                                  )

                self.rid_list.append(row['slice_num'])
                self.subject_list.append(subject)
                self.subject_dict[row['slice_num']] = subject


    @property
    def image_shape(self):
        ishape = self.subject_list[0].image_shape
        # for sbj in self.subject_list[1:]:
        #     sshape = sbj.image_shape
        #     ishape = tuple([max(a,b) for a,b in zip(ishape, sshape)])
        return ishape

    def __len__(self):
        return len(self.subject_list)

