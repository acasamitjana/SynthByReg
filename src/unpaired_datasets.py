# py
import random
import time

# third party imports
import torch
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_dilation
from skimage.exposure import match_histograms
import numpy as np
import pdb

#project imports
from src.utils import image_transforms as tf
from src.utils.image_utils import one_hot_encoding

class RegistrationDataset3D(Dataset):
    def __init__(self, data_loader_ref, data_loader_flo, affine_params, nonlinear_params, tf_params=None,
                 da_params=None, norm_params=None, mask_dilation=None, to_tensor=True,  num_classes=False, train=True):
        '''

        :param data_loader:
        :param rotation_params:
        :param nonlinear_params:
        :param tf_params:
        :param da_params:
        :param norm_params:
        :param hist_match:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param num_classes: (int) number of classes for one-hot encoding. If num_classes=-1, one-hot is not performed.
        :param train:
        '''

        self.data_loader_Ref = data_loader_ref
        self.data_loader_flo = data_loader_flo
        self.subject_list_ref = data_loader_ref.subject_list
        self.subject_list_flo = data_loader_ref.subject_list_flo
        self.N = len(self.subject_list_flo)
        self.to_tensor = to_tensor

        self.tf_params = tf.Compose(tf_params) if tf_params is not None else None
        self.da_params = tf.Compose_DA(da_params) if da_params is not None else None
        self.norm_params = norm_params if norm_params is not None else lambda x: x
        self.mask_dilation = mask_dilation

        self.affine_params = affine_params
        self.nonlinear_params = nonlinear_params
        self.num_classes = num_classes

        self._image_shape = self.tf_params._compute_data_shape(data_loader_ref.image_shape) if tf_params is not None else data_loader_ref.image_shape
        self.n_dims = data_loader_ref.n_dims
        self.n_channels = data_loader_ref.n_channels
        self.train = train

    def mask_image(self, image, mask, mask_value=None):

        if self.norm_params is not None and mask_value is None:
            mask_value = self.norm_params.get_mask_value(image)
        else:
            mask_value = 0

        mask_bool = mask > 0

        image[~mask_bool] = mask_value

        return image

    def get_data(self, slice, *args, **kwargs):

        x = slice.load_data(*args, **kwargs)
        x_mask = slice.load_mask(*args, **kwargs)
        x_labels = slice.load_labels(*args, **kwargs)
        x = self.mask_image(x, x_mask, mask_value=0)

        return x, x_mask, x_labels


    def get_intermodal_data(self, slice_ref, slice_flo, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels = self.get_data(slice_ref, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels = self.get_data(slice_flo, *args, **kwargs)
        x_flo_orig = x_flo

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels,'x_flo_labels': x_flo_labels,
        }

        return data_dict

    def get_intramodal_data(self, slice_ref, slice_flo, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels = self.get_data(slice_ref, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels = self.get_data(slice_flo, *args, **kwargs)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels,

        }

        return data_dict

    def get_deformation_field(self, num=2):
        affine_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            affine = self.affine_params.get_affine(self._image_shape)

            nlf_x, nlf_y, nlf_z = self.nonlinear_params.get_lowres_strength(ndim=3)
            nonlinear_field = np.zeros((3,) + nlf_x.shape)
            nonlinear_field[0] = nlf_x
            nonlinear_field[1] = nlf_y
            nonlinear_field[2] = nlf_z

            affine_list.append(affine)
            nonlinear_field_list.append(nonlinear_field)

        return affine_list, nonlinear_field_list

    def data_augmentation(self, data_dict):
        x_ref = data_dict['x_ref']
        x_ref_mask = data_dict['x_ref_mask']
        x_ref_labels = data_dict['x_ref_labels']
        x_flo = data_dict['x_flo']
        x_flo_mask = data_dict['x_flo_mask']
        x_flo_labels = data_dict['x_flo_labels']

        if self.da_params is not None:
            img = self.da_params([x_ref, x_ref_mask, x_ref_labels], mask_flag=[False, True, True])
            x_ref, x_ref_mask, x_ref_labels = img
            x_ref[np.isnan(x_ref)] = 0
            x_ref_mask[np.isnan(x_ref_mask)] = 0
            x_ref_labels[np.isnan(x_ref_labels)] = 0

            img = self.da_params([x_flo, x_flo_mask, x_flo_labels], mask_flag=[False, True, True])
            x_flo, x_flo_mask, x_flo_labels = img
            x_flo[np.isnan(x_flo)] = 0
            x_flo_mask[np.isnan(x_flo_mask)] = 0
            x_flo_labels[np.isnan(x_flo_labels)] = 0


        if self.mask_dilation is not None:
            x_ref_mask = binary_dilation(x_ref_mask, structure=self.mask_dilation)
            x_flo_mask = binary_dilation(x_flo_mask, structure=self.mask_dilation)

        data_dict['x_ref'] = x_ref
        data_dict['x_ref_mask'] = x_ref_mask
        data_dict['x_ref_labels'] = x_ref_labels
        data_dict['x_flo'] = x_flo
        data_dict['x_flo_mask'] = x_flo_mask
        data_dict['x_flo_labels'] = x_flo_labels

        return data_dict

    def convert_to_tensor(self, data_dict):

        # u1 = np.unique(data_dict['x_ref_labels'])
        # u2 = np.unique(data_dict['x_flo_labels'])
        # categories = list(set(u1) & set(u2))

        for k, v in data_dict.items():
            if 'landmarks' in k:
                continue
            elif 'labels' in k and self.num_classes:
                v = one_hot_encoding(v, self.num_classes)#, categories=categories)
                data_dict[k] = torch.from_numpy(v).float()
            elif isinstance(v, list):
                data_dict[k] = [torch.from_numpy(vl).float() for vl in v]
            else:
                data_dict[k] = torch.from_numpy(v[np.newaxis]).float()

        return data_dict

    def __getitem__(self, index):

        index_ref = np.random.randint(0, len(self.subject_list_ref), size=1)
        subject_ref = self.subject_list_ref[index_ref]
        subject_flo = self.subject_list_flo[index]
        rid_ref = subject_ref.id
        rid_flo = subject_flo.id

        data_dict = self.get_intermodal_data(subject_ref, subject_flo)
        data_dict['x_ref_init_mask'] = data_dict['x_ref_mask']
        data_dict['x_flo_init_mask'] = data_dict['x_flo_mask']
        data_dict = self.data_augmentation(data_dict)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)
        data_dict['rid'] = rid_ref + '_' + rid_flo

        for k, v in data_dict.items():
            if v is None:
                print(k)
        return data_dict

    @property
    def image_shape(self):
        return self._image_shape


    def __len__(self):
        return self.N

