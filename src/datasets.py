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

class RegistrationDataset2D(Dataset):
    def __init__(self, data_loader, affine_params, nonlinear_params, tf_params=None, da_params=None, norm_params=None,
                 hist_match=False, mask_dilation=None, to_tensor=True, landmarks=False, num_classes=-1,
                 sbj_per_epoch=1, train=True):
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

        self.data_loader = data_loader
        self.subject_list = data_loader.subject_list
        self.N = len(self.subject_list)
        self.to_tensor = to_tensor

        self.landmarks = landmarks

        self.tf_params = tf.Compose(tf_params) if tf_params is not None else None
        self.da_params = tf.Compose_DA(da_params) if da_params is not None else None
        self.norm_params = norm_params if norm_params is not None else lambda x: x
        self.mask_dilation = mask_dilation

        self.hist_match = hist_match
        self.affine_params = affine_params
        self.nonlinear_params = nonlinear_params
        self.num_classes = num_classes

        self._image_shape = self.tf_params._compute_data_shape(data_loader.image_shape) if tf_params is not None else data_loader.image_shape
        self.n_dims = data_loader.n_dims
        self.n_channels = data_loader.n_channels
        self.train = train
        self.sbj_per_epoch = sbj_per_epoch

        self.ref_modality = None
        self.flo_modality = None

    def mask_image(self, image, mask, mask_value=None):
        ndim = len(image.shape)

        if self.norm_params is not None and mask_value is None:
            mask_value = self.norm_params.get_mask_value(image)
        else:
            mask_value = 0

        mask_bool = mask > 0

        if ndim == 3:
            for it_z in range(image.shape[-1]):
                image_tmp = image[..., it_z]
                image_tmp[~mask_bool] = mask_value
                image[..., it_z] = image_tmp
        else:
            image[~mask_bool] = mask_value

        return image

    def get_ref_data(self, slice, *args, **kwargs):

        if self.ref_modality == 'MRI':
            x_ref = slice.load_mri(*args, **kwargs)
            x_ref_mask = slice.load_mri_mask(*args, **kwargs)
            x_ref_labels = slice.load_mri_labels(*args, **kwargs)
            x_ref_landmarks = slice.load_mri_landmarks_affine() if self.landmarks is True else None
            x_ref = self.mask_image(x_ref,x_ref_mask, mask_value=0)

        elif self.ref_modality == 'HISTO':
            x_ref = slice.load_histo_affine(*args, **kwargs)
            x_ref_mask = slice.load_histo_mask_affine(*args, **kwargs)
            x_ref_labels = slice.load_histo_labels_affine(*args, **kwargs)
            x_ref_landmarks = slice.load_histo_landmarks_affine() if self.landmarks is True else None
            x_ref = self.mask_image(x_ref, x_ref_mask, mask_value=0)

        else:
            raise ValueError("Please, specify a valid reference modality")

        if np.sum(x_ref_mask) > 0:
            x_ref = self.norm_params(x_ref, x_ref_mask)
            # x_ref = self.mask_image(x_ref, x_ref_mask, mask_value=0)

        return x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks

    def get_flo_data(self, slice, *args, **kwargs):

        if self.flo_modality == 'MRI':
            x_flo = slice.load_mri(*args, **kwargs)
            x_flo_mask = slice.load_mri_mask(*args, **kwargs)
            x_flo_labels = slice.load_mri_labels(*args, **kwargs)
            x_flo_landmarks = slice.load_mri_landmarks_affine() if self.landmarks is True else None
            x_flo = self.mask_image(x_flo, x_flo_mask, mask_value=0)

        elif self.flo_modality == 'HISTO':
            x_flo = slice.load_histo_affine(*args, **kwargs)
            x_flo_mask = slice.load_histo_mask_affine(*args, **kwargs)
            x_flo_labels = slice.load_histo_labels_affine(*args, **kwargs)
            x_flo_landmarks = slice.load_histo_landmarks_affine() if self.landmarks is True else None
            x_flo = self.mask_image(x_flo, x_flo_mask, mask_value=0)

        else:
            raise ValueError("Please, specify a valid floating modality")

        if np.sum(x_flo_mask) > 0:
            x_flo = self.norm_params(x_flo, x_flo_mask)
            x_flo = self.mask_image(x_flo, x_flo_mask)

        return x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks

    def get_intermodal_data(self, slice, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks = self.get_ref_data(slice, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks = self.get_flo_data(slice, *args, **kwargs)
        x_flo_orig = x_flo

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0) and self.flo_modality != 'MRI':
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img


        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels,'x_flo_labels': x_flo_labels,
        }

        if self.landmarks:
            data_dict['x_ref_landmarks'] = x_ref_landmarks
            data_dict['x_flo_landmarks'] = x_flo_landmarks

        return data_dict

    def get_intramodal_data(self, slice_ref, slice_flo, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks = self.get_ref_data(slice_ref, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks = self.get_flo_data(slice_flo, *args, **kwargs)
        x_flo_orig = x_flo

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0) and self.flo_modality != 'MRI':
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels,

        }

        if self.landmarks:
            data_dict['x_ref_landmarks'] = x_ref_landmarks
            data_dict['x_flo_landmarks'] = x_flo_landmarks

        return data_dict

    def get_deformation_field(self, num=2):
        affine_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            affine = self.affine_params.get_affine(self._image_shape)
            nlf_x, nlf_y = self.nonlinear_params.get_lowres_strength(ndim=2)
            nonlinear_field = np.zeros((2,) + nlf_x.shape)
            nonlinear_field[0] = nlf_y
            nonlinear_field[1] = nlf_x

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

        for k, v in data_dict.items():
            if 'landmarks' in k:
                continue
            elif 'labels' in k and self.num_classes:
                v = one_hot_encoding(v, self.num_classes)
                data_dict[k] = torch.from_numpy(v).float()
            elif isinstance(v, list):
                data_dict[k] = [torch.from_numpy(vl).float() for vl in v]
            else:
                data_dict[k] = torch.from_numpy(v[np.newaxis]).float()

        return data_dict

    def __len__(self):
        return self.sbj_per_epoch * self.N

class InterModalRegistrationDataset2D(RegistrationDataset2D):
    '''
    Class for intermodal registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, rotation_params, nonlinear_params, ref_modality, flo_modality, tf_params=None,
                 da_params=None, norm_params=None, hist_match=False, mask_dilation=None, landmarks=False, train=True,
                 num_classes=False, to_tensor=True, sbj_per_epoch=1):

        super().__init__(data_loader, rotation_params, nonlinear_params, tf_params=tf_params, da_params=da_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.ref_modality = ref_modality
        self.flo_modality = flo_modality
        self.hist_match = hist_match

    def __getitem__(self, index):

        subject = self.subject_list[index]

        if hasattr(subject, 'get_slice'):
            slice, slice_num = subject.get_slice(random_seed=None if self.train else 44)
            rid = slice.id
        else:
            slice = subject
            rid = subject.id

        data_dict = self.get_intermodal_data(slice)
        data_dict['x_ref_init_mask'] = data_dict['x_ref_mask']
        data_dict['x_flo_init_mask'] = data_dict['x_flo_mask']
        data_dict = self.data_augmentation(data_dict)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)
        data_dict['rid'] = rid

        for k,v in data_dict.items():
            if v is None:
                print(k)
        return data_dict

    @property
    def image_shape(self):
        return self._image_shape

class IntraModalRegistrationDataset2D(RegistrationDataset2D):
    '''
    Basic class for registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, rotation_params, nonlinear_params, modality, tf_params=None, da_params=None,
                 norm_params=None, hist_match=False, mask_dilation=None, to_tensor=True, landmarks=False, train=True,
                 num_classes=False, sbj_per_epoch=1, neighbor_distance=-1, fix_neighbors=False):

        '''
        :param data_loader:
        :param transform_parameters:
        :param data_augmentation_parameters:
        :param normalization:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param train:
        :param modality:
        :param neighbor_distance: forward distance in mm (positive) or in number of neighbors (negative)
        :param reverse: it only applies when neighbor_distance<0
        '''
        super().__init__(data_loader, rotation_params, nonlinear_params, tf_params=tf_params, da_params=da_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.neighbor_distance = neighbor_distance
        self.ref_modality = modality
        self.flo_modality = modality
        self.fix_neighbors = fix_neighbors
        self.reverse = False

    def get_slice(self, subject_ref, index_ref):

        if hasattr(subject_ref, 'get_slice'):
            slice_ref, index_ref = subject_ref.get_slice(random_seed=None if self.train else 44)
            rid_ref = slice_ref.id
            slice_list = subject_ref.slice_list

        else:
            slice_ref = subject_ref
            rid_ref = subject_ref.id
            slice_list = self.subject_list

        if self.neighbor_distance == 0:
            slice_flo = slice_ref
            rid_flo = rid_ref
            index_flo = index_ref

            return slice_ref, slice_flo, rid_ref, rid_flo, index_flo

        if not self.fix_neighbors: # here we get randomly any neighbor
            if self.neighbor_distance > 0:
                slices_available = [sref for sref in slice_list if
                                    sref.id_mm <= slice_ref.id_mm + self.neighbor_distance and
                                    sref.id_mm >= slice_ref.id_mm - self.neighbor_distance]

                index_flo = int(np.random.choice(len(slices_available), size=1))
                slice_flo = slices_available[index_flo]
                rid_flo = slice_flo.id

            else:
                idx_min = index_ref + self.neighbor_distance if index_ref + self.neighbor_distance > 0 else 0
                idx_max = index_ref - self.neighbor_distance
                slices_available = slice_list[idx_min:idx_max]
                index_flo = int(np.random.choice(len(slices_available), size=1))
                slice_flo = slices_available[index_flo]
                rid_flo = slice_flo.id

        else:
            if self.neighbor_distance > 0:
                index_mm = slice_ref.id_mm + self.neighbor_distance
                index_mm_available = np.asarray([sref.id_mm for sref in slice_list])
                index_flo = np.argsort(np.abs(index_mm_available - index_mm))[0]
            else:
                index_flo = index_ref - self.neighbor_distance
                if index_flo >= len(slice_list):  # if it reached the last position, register the other way around.
                    index_flo = index_ref + self.neighbor_distance

            slice_flo = slice_list[index_flo]
            rid_ref = slice_ref.id
            rid_flo = slice_flo.id

        return slice_ref, slice_flo, rid_ref, rid_flo, index_flo

    def __getitem__(self, index):

        subject_ref = self.subject_list[index]
        slice_ref, slice_flo, rid_ref, rid_flo, _ = self.get_slice(subject_ref, index)
        rid = rid_flo + '_to_' + rid_ref


        data_dict = self.get_intramodal_data(slice_ref, slice_flo)
        data_dict = self.data_augmentation(data_dict)
        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)
        data_dict['rid'] = rid

        return data_dict

    @property
    def image_shape(self):
        return self._image_shape

    def __len__(self):
        if self.neighbor_distance < 0 and self.fix_neighbors:
            return super().__len__() + self.neighbor_distance
        else:
            return super().__len__()


class RegistrationDataset3D(Dataset):
    def __init__(self, data_loader, affine_params, nonlinear_params, tf_params=None, da_params=None, norm_params=None,
                 hist_match=False, mask_dilation=None, to_tensor=True, landmarks=False, num_classes=-1,
                 sbj_per_epoch=1, train=True):
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

        self.data_loader = data_loader
        self.subject_list = data_loader.subject_list
        self.N = len(self.subject_list)
        self.to_tensor = to_tensor

        self.landmarks = landmarks

        self.tf_params = tf.Compose(tf_params) if tf_params is not None else None
        self.da_params = tf.Compose_DA(da_params) if da_params is not None else None
        self.norm_params = norm_params if norm_params is not None else lambda x: x
        self.mask_dilation = mask_dilation

        self.hist_match = hist_match
        self.affine_params = affine_params
        self.nonlinear_params = nonlinear_params
        self.num_classes = num_classes

        self._image_shape = self.tf_params._compute_data_shape(data_loader.image_shape) if tf_params is not None else data_loader.image_shape
        self.n_dims = data_loader.n_dims
        self.n_channels = data_loader.n_channels
        self.train = train
        self.sbj_per_epoch = sbj_per_epoch

        self.ref_modality = None
        self.flo_modality = None

    def mask_image(self, image, mask, mask_value=None):

        if self.norm_params is not None and mask_value is None:
            mask_value = self.norm_params.get_mask_value(image)
        else:
            mask_value = 0

        mask_bool = mask > 0

        image[~mask_bool] = mask_value

        return image

    def get_ref_data(self, slice, *args, **kwargs):

        if self.ref_modality == 'MRI':
            x_ref = slice.load_mri(*args, **kwargs)
            x_ref_mask = slice.load_mri_mask(*args, **kwargs)
            x_ref_labels = slice.load_mri_labels(*args, **kwargs)
            x_ref_landmarks = slice.load_mri_landmarks_affine() if self.landmarks is True else None
            x_ref = self.mask_image(x_ref,x_ref_mask, mask_value=0)

        elif self.ref_modality == 'HISTO':
            x_ref = slice.load_histo_affine(*args, **kwargs)
            x_ref_mask = slice.load_histo_mask_affine(*args, **kwargs)
            x_ref_labels = slice.load_histo_labels_affine(*args, **kwargs)
            x_ref_landmarks = slice.load_histo_landmarks_affine() if self.landmarks is True else None
            x_ref = self.mask_image(x_ref, x_ref_mask, mask_value=0)

        else:
            raise ValueError("Please, specify a valid reference modality")

        if np.sum(x_ref_mask) > 0:
            x_ref = self.norm_params(x_ref, x_ref_mask)
            # x_ref = self.mask_image(x_ref, x_ref_mask, mask_value=0)

        return x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks

    def get_flo_data(self, slice, *args, **kwargs):

        if self.flo_modality == 'MRI':
            x_flo = slice.load_mri(*args, **kwargs)
            x_flo_mask = slice.load_mri_mask(*args, **kwargs)
            x_flo_labels = slice.load_mri_labels(*args, **kwargs)
            x_flo_landmarks = slice.load_mri_landmarks_affine() if self.landmarks is True else None
            x_flo = self.mask_image(x_flo, x_flo_mask, mask_value=0)

        elif self.flo_modality == 'HISTO':
            x_flo = slice.load_histo_affine(*args, **kwargs)
            x_flo_mask = slice.load_histo_mask_affine(*args, **kwargs)
            x_flo_labels = slice.load_histo_labels_affine(*args, **kwargs)
            x_flo_landmarks = slice.load_histo_landmarks_affine() if self.landmarks is True else None
            x_flo = self.mask_image(x_flo, x_flo_mask, mask_value=0)

        else:
            raise ValueError("Please, specify a valid floating modality")

        if np.sum(x_flo_mask) > 0:
            x_flo = self.norm_params(x_flo, x_flo_mask)
            # x_flo = self.mask_image(x_flo, x_flo_mask, mask_value=0)

        return x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks

    def get_intermodal_data(self, slice, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks = self.get_ref_data(slice, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks = self.get_flo_data(slice, *args, **kwargs)
        x_flo_orig = x_flo

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0) and self.flo_modality != 'MRI':
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img


        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels,'x_flo_labels': x_flo_labels,
        }

        if self.landmarks:
            data_dict['x_ref_landmarks'] = x_ref_landmarks
            data_dict['x_flo_landmarks'] = x_flo_landmarks

        return data_dict

    def get_intramodal_data(self, slice_ref, slice_flo, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks = self.get_ref_data(slice_ref, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks = self.get_flo_data(slice_flo, *args, **kwargs)
        x_flo_orig = x_flo

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0) and self.flo_modality != 'MRI':
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels,

        }

        if self.landmarks:
            data_dict['x_ref_landmarks'] = x_ref_landmarks
            data_dict['x_flo_landmarks'] = x_flo_landmarks

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

    def __len__(self):
        return self.sbj_per_epoch * self.N

class InterModalRegistrationDataset3D(RegistrationDataset3D):
    '''
    Class for intermodal registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, rotation_params, nonlinear_params, ref_modality, flo_modality, tf_params=None,
                 da_params=None, norm_params=None, hist_match=False, mask_dilation=None, landmarks=False, train=True,
                 num_classes=False, to_tensor=True, sbj_per_epoch=1):

        super().__init__(data_loader, rotation_params, nonlinear_params, tf_params=tf_params, da_params=da_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.ref_modality = ref_modality
        self.flo_modality = flo_modality
        self.hist_match = hist_match

    def __getitem__(self, index):

        subject = self.subject_list[index]

        if hasattr(subject, 'get_slice'):
            slice, slice_num = subject.get_slice(random_seed=None if self.train else 44)
            rid = slice.id
        else:
            slice = subject
            rid = subject.id

        data_dict = self.get_intermodal_data(slice)
        data_dict['x_ref_init_mask'] = data_dict['x_ref_mask']
        data_dict['x_flo_init_mask'] = data_dict['x_flo_mask']
        data_dict = self.data_augmentation(data_dict)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)
        data_dict['rid'] = rid

        for k,v in data_dict.items():
            if v is None:
                print(k)
        return data_dict

    @property
    def image_shape(self):
        return self._image_shape

class IntraModalRegistrationDataset3D(RegistrationDataset3D):
    '''
    Basic class for registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, rotation_params, nonlinear_params, modality, tf_params=None, da_params=None,
                 norm_params=None, hist_match=False, mask_dilation=None, to_tensor=True, landmarks=False, train=True,
                 num_classes=False, sbj_per_epoch=1, neighbor_distance=-1, fix_neighbors=False):

        '''
        :param data_loader:
        :param transform_parameters:
        :param data_augmentation_parameters:
        :param normalization:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param train:
        :param modality:
        :param neighbor_distance: forward distance in mm (positive) or in number of neighbors (negative)
        :param reverse: it only applies when neighbor_distance<0
        '''
        super().__init__(data_loader, rotation_params, nonlinear_params, tf_params=tf_params, da_params=da_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.neighbor_distance = neighbor_distance
        self.ref_modality = modality
        self.flo_modality = modality
        self.fix_neighbors = fix_neighbors
        self.reverse = False

    def get_slice(self, subject_ref, index_ref):

        if hasattr(subject_ref, 'get_slice'):
            slice_ref, index_ref = subject_ref.get_slice(random_seed=None if self.train else 44)
            rid_ref = slice_ref.id
            slice_list = subject_ref.slice_list

        else:
            slice_ref = subject_ref
            rid_ref = subject_ref.id
            slice_list = self.subject_list

        if self.neighbor_distance == 0:
            slice_flo = slice_ref
            rid_flo = rid_ref
            index_flo = index_ref

        else:
            index_flo = int(np.random.choice(len(slice_list), size=1))
            slice_flo = slice_list[index_flo]
            rid_flo = slice_flo.id

        return slice_ref, slice_flo, rid_ref, rid_flo, index_flo

    def __getitem__(self, index):

        subject_ref = self.subject_list[index]
        slice_ref, slice_flo, rid_ref, rid_flo, _ = self.get_slice(subject_ref, index)
        rid = rid_flo + '_to_' + rid_ref

        data_dict = self.get_intramodal_data(slice_ref, slice_flo)
        data_dict = self.data_augmentation(data_dict)
        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)
        data_dict['rid'] = rid

        return data_dict

    @property
    def image_shape(self):
        return self._image_shape

    def __len__(self):
        if self.neighbor_distance < 0 and self.fix_neighbors:
            return super().__len__() + self.neighbor_distance
        else:
            return super().__len__()


class InterModalLabelRegistrationDataset(object):
    '''
    Basic class for registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, transform_parameters=None, data_augmentation_parameters=None, normalization=None,
                 mask_dilation=False, to_tensor=True, landmarks = False, train=True, ref_modality= 'MRI',
                 flo_modality= 'HISTO'):

        self.subject_list = data_loader.subject_list
        self.N = len(self.subject_list)
        self.to_tensor = to_tensor

        self.landmarks=landmarks

        self.transform = tf.Compose(transform_parameters)
        self.data_augmentation = tf.Compose_DA(data_augmentation_parameters)
        self.normalization = normalization if normalization is not None else lambda x: x
        self.mask_dilation = mask_dilation

        self._image_shape = self.transform._compute_data_shape(data_loader.image_shape)
        self.n_dims = data_loader.n_dims
        self.n_channels = data_loader.n_channels
        self.train = train

        self.ref_modality = ref_modality
        self.flo_modality = flo_modality


    def __getitem__(self, index):

        subject = self.subject_list[index]
        rid = subject.id

        if hasattr(subject, 'get_slice'):
            slice = subject.get_slice(random_seed=np.mod(torch.utils.data.get_worker_info().seed,2**16-1))
            rid = slice.id

        else:

            slice = subject


        x_ref = slice.load_mri_labels()
        x_ref_landmarks = slice.load_mri_landmarks_affine() if self.landmarks is True else None
        x_flo = slice.load_LFB_labels()
        x_flo_landmarks = slice.load_LFB_landmarks_affine() if self.landmarks is True else None

        x_ref, x_flo = self.transform([x_ref, x_flo])

        x_ref = self.data_augmentation(x_ref, mask_flag=False)
        x_ref[np.isnan(x_ref)]=0

        x_flo = self.data_augmentation(x_flo, mask_flag=False)
        x_flo[np.isnan(x_flo)]=0

        if self.to_tensor:
            x_ref, x_flo = torch.from_numpy(x_ref[np.newaxis]).float(), torch.from_numpy(x_flo[np.newaxis]).float()

        if self.landmarks:
            return x_ref, x_flo, x_ref_landmarks, x_flo_landmarks, rid
        else:
            return x_ref, x_flo, rid

    def __len__(self):
        return self.N

    @property
    def image_shape(self):
        return self._image_shape

