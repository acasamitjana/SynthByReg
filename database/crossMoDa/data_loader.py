import csv
from os.path import join
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_opening


class Subject():
    def __init__(self, id, data_path, id_num):

        self.data_path = data_path
        self._id = id
        self._id_num = id_num

        self.image_mri = join(data_path, 'MR', 'images', 'img' + id + '_MR.nii.gz')
        self.mask_mri = join(data_path, 'MR', 'masks', 'mask' + id + '_MR.nii.gz')
        self.seg_mri = join(data_path, 'MR', 'segmentation', 'seg' + id + '_MR.nii.gz')

        self.image_ct = join(data_path, 'CT', 'images', 'img' + id + '_CT.nii.gz')
        self.mask_ct = join(data_path, 'CT', 'masks', 'mask' + id + '_CT.nii.gz')
        self.seg_ct = join(data_path, 'CT', 'segmentation', 'seg' + id + '_CT.nii.gz')

        self.matrix_ct_affine = join(data_path, 'CT_affine', 'mat' + id + '_CT.txt')
        self.image_ct_affine = join(data_path, 'CT_affine', 'images', 'img' + id + '_CT.nii.gz')
        self.mask_ct_affine = join(data_path, 'CT_affine', 'masks', 'mask' + id + '_CT.nii.gz')
        self.seg_ct_affine = join(data_path, 'CT_affine', 'segmentation', 'seg' + id + '_CT.nii.gz')


    @property
    def id(self):
        return self._id

    @property
    def id_mm(self):
        return self._id_num

    @property
    def vox2ras(self):
        proxy = nib.load(self.image_mri)
        return proxy.affine

    @property
    def image_shape(self):
        proxy = nib.load(self.image_mri)
        return proxy.shape

    def load_mri(self, *args, **kwargs):
        proxy = nib.load(self.image_mri)
        return np.asarray(proxy.dataobj)

    def load_histo(self,*args, **kwargs):
        proxy = nib.load(self.image_ct)
        return np.asarray(proxy.dataobj)

    def load_histo_affine(self, *args, **kwargs):
        proxy = nib.load(self.image_ct_affine)
        return np.asarray(proxy.dataobj)

    def load_mri_mask(self, *args, **kwargs):
        proxy = nib.load(self.mask_mri)
        return np.asarray(proxy.dataobj)

    def load_histo_mask(self, *args, **kwargs):
        proxy = nib.load(self.mask_ct)
        return np.asarray(proxy.dataobj)

    def load_histo_mask_affine(self, *args, **kwargs):
        proxy = nib.load(self.mask_ct_affine)
        return np.asarray(proxy.dataobj)

    def load_mri_labels(self, *args, **kwargs):
        proxy = nib.load(self.seg_mri)
        return np.asarray(proxy.dataobj)

    def load_histo_labels(self, *args, **kwargs):
        proxy = nib.load(self.seg_ct)
        return np.asarray(proxy.dataobj)

    def load_histo_labels_affine(self, *args, **kwargs):
        proxy = nib.load(self.seg_ct_affine)
        return np.asarray(proxy.dataobj)

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
        self.data_path = database_config['DATA_DIR']
        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, rid_list=None, **kwargs):
        self.n_dims = 3
        self.n_channels = 1

        self.subject_list = []
        self.rid_list = []
        self.subject_dict = {}

        with open(self.database_config['DATA_FILE'], 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                rid = row['Name']
                if rid_list is not None:
                    if rid not in rid_list:
                        continue

                subject = Subject(id=rid,
                                  data_path=self.database_config['DATA_DIR'],
                                  id_num=int(rid.split('_')[0])
                                  )

                self.rid_list.append(rid)
                self.subject_list.append(subject)
                self.subject_dict[rid] = subject


    @property
    def image_shape(self):
        ishape = self.subject_list[0].image_shape
        # for sbj in self.subject_list[1:]:
        #     sshape = sbj.image_shape
        #     ishape = tuple([max(a,b) for a,b in zip(ishape, sshape)])
        return ishape

    def __len__(self):
        return len(self.subject_list)

