import csv
from collections import OrderedDict
from os import listdir
from os.path import join, exists

import numpy as np
from PIL import Image


class Subject():
    def __init__(self, id, data_path, mri_dir, histo_dir, landmarks_dir):

        self.data_path = data_path
        self._id = id

        self.image_mri = join(mri_dir, 'images', 'slice_' + str(id) + '.png')
        # self.labels_mri = join(mri_dir, 'labels_tif', 'slice_' + str(id) + '.tif')
        self.labels_mri = join(mri_dir, 'labels', 'slice_' + str(id) + '.png')
        self.image_histo = join(histo_dir, 'images', 'slice_' + str(id) + '.png')
        self.labels_histo = join(histo_dir, 'labels', 'slice_' + str(id) + '.png')
        self.image_histo_affine = join(histo_dir, 'images_affine', 'slice_' + str(id) + '.png')
        self.labels_histo_affine = join(histo_dir, 'labels_affine', 'slice_' + str(id) + '.png')
        self.matrix_histo_affine = join(histo_dir, 'affine', 'slice_' + str(id) + '.txt')
        self.landmarks = join(landmarks_dir, 'slice_' + str(id) + '.txt')


    @property
    def id(self):
        return self._id

    @property
    def id_mm(self):
        return int(self._id)

    @property
    def image_shape(self):
        mask = self.load_mri()
        return mask.shape

    def load_mri(self,):
        data = Image.open(self.image_mri)
        return np.array(data)

    def load_mri_mask(self):
        labels = Image.open(self.labels_mri)
        return (np.array(labels) > 0).astype('uint8')

    def load_mri_labels(self):
        return self.load_mri_mask()

    def load_histo(self,):
        data = Image.open(self.image_histo)
        return np.array(data)

    def load_histo_mask(self):
        labels = Image.open(self.labels_histo)
        return (np.array(labels) > 0).astype('uint8')

    def load_histo_labels(self):
        return self.load_histo_mask()

    def load_histo_affine(self, ):
        data = Image.open(self.image_histo_affine)
        return np.array(data)

    def load_histo_mask_affine(self):
        labels = Image.open(self.labels_histo_affine)
        return (np.array(labels) > 0).astype('uint8')

    def load_histo_labels_affine(self):
        return self.load_histo_mask_affine()

    def load_mri_landmarks(self):
        landmarks_list = []
        with open(self.landmarks, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_list.append((float(row[0]), float(row[1])))

        return landmarks_list

    def load_mri_landmarks_affine(self):
        return self.load_mri_landmarks()

    def load_histo_landmarks(self):
        landmarks_list = []
        with open(self.landmarks, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_list.append((float(row[2]), float(row[3])))

        return landmarks_list

    def load_histo_landmarks_affine(self):
        landmarks_list = self.load_histo_landmarks()
        affine_matrix = np.linalg.inv(self.load_affine_matrix(full=True))

        landmarks_affine_list = []
        for l in landmarks_list:
            al = affine_matrix @ np.asarray(list(l) + [0, 1])
            landmarks_affine_list.append(tuple(al[:2]))

        return landmarks_affine_list

    def load_landmarks(self):
        landmarks_dict = OrderedDict({'histo': {'X': [], 'Y': []}, 'mri': {'X': [], 'Y': []}})

        with open(self.landmarks, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_dict['histo']['X'].append(float(row[2]))
                landmarks_dict['histo']['Y'].append(float(row[3]))
                landmarks_dict['mri']['X'].append(float(row[0]))
                landmarks_dict['mri']['Y'].append(float(row[1]))

        return landmarks_dict

    def load_affine_matrix(self, full=False, *args, **kwargs):

        with open(self.matrix_histo_affine, 'r') as csvfile:
            if full:
                affine_matrix = np.zeros((4, 4))
                csvreader = csv.reader(csvfile, delimiter=' ')
                for it_row, row in enumerate(csvreader):
                    affine_matrix[it_row, 0] = float(row[0])
                    affine_matrix[it_row, 1] = float(row[1])
                    affine_matrix[it_row, 2] = float(row[2])
                    affine_matrix[it_row, 3] = float(row[3])
            else:
                affine_matrix = np.zeros((2, 3))

                csvreader = csv.reader(csvfile, delimiter=' ')
                row = next(csvreader)
                affine_matrix[0, 0] = float(row[0])
                affine_matrix[0, 1] = float(row[1])
                affine_matrix[0, 2] = float(row[3])
                row = next(csvreader)
                affine_matrix[1, 0] = float(row[0])
                affine_matrix[1, 1] = float(row[1])
                affine_matrix[1, 2] = float(row[3])

        return affine_matrix

class DataLoader(object):

    def __init__(self, database_config, *args, **kwargs):
        self.database_config = database_config
        self.data_path = database_config['BASE_DIR']
        self._initialize_dataset(*args, **kwargs)

        self.n_dims = 2
        self.n_channels = 1

    def _initialize_dataset(self, rid_list=None):

        self.subject_dict = {}
        with open(join(self.database_config['BASE_DIR'], self.database_config['DATA_FILE']), 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if rid_list is not None:
                    if row['SLICE_ID'] not in rid_list:
                        continue
                self.rid_list.append(row['SLICE_ID'])

                self.subject_dict[row['SLICE_ID']] = Subject(row['SLICE_ID'],
                                                             self.database_config['BASE_DIR'],
                                                             self.database_config['MRI_DIR'],
                                                             self.database_config['HISTOLOGY_DIR'],
                                                             self.database_config['LANDMARKS_DIR']
                                                             )

    def __len__(self):
        return len(self.subject_dict.keys())

    @property
    def rid_list(self):
        return list(self.subject_dict.keys())

    @property
    def subject_list(self):
        return list(self.subject_dict.values())


    @property
    def image_shape(self):
        ishape = self.subject_list[0].image_shape
        for sbj in self.subject_list[1:]:
            sshape = sbj.image_shape
            ishape = tuple([max(a,b) for a,b in zip(ishape, sshape)])
        return ishape


    #
    # def load_histo(self, nparray = False):
    #     images = []
    #     images_labels = []
    #     images_dict = {'num_channels': 0, 'unique_labels': None}
    #
    #     num_images = 0
    #     for slice_id, files in self.slices_dict.items():
    #         if files['histo_image'] != '':
    #             num_images += 1
    #             histo_image = Image.open(files['histo_image'])
    #             if 'L' not in histo_image.mode:
    #                 histo_image = histo_image.convert('L')
    #
    #             images_dict['num_channels'] = histo_image.size if len(histo_image.size) > 2 else 1
    #             images_dict['image_shape'] = histo_image.size
    #
    #             if nparray:
    #                 images.append(np.array(histo_image))
    #             else:
    #                 images.append(histo_image)
    #
    #
    #
    #         if files['histo_labels'] != '':
    #             histo_labels = Image.open(files['histo_labels'])
    #             if 'L' not in histo_labels.mode:
    #                 histo_labels = histo_labels.convert('L')
    #
    #             if images_dict['unique_labels'] is None:
    #                 images_dict['unique_labels'] = np.unique(histo_labels)
    #             else:
    #                 images_dict['unique_labels'] = list(set().union(np.unique(histo_labels), images_dict['unique_labels']))
    #
    #
    #
    #             if nparray:
    #                 images_labels.append(np.asarray(histo_labels))
    #             else:
    #                 images_labels.append(histo_labels)
    #
    #
    #     if nparray:
    #         images = np.asarray(images)
    #         images_labels = np.asarray(images_labels)
    #
    #     images_dict['num_images'] = num_images
    #     return images, images_labels, images_dict
    #
    #
    # def load_mri(self, nparray=False):
    #     images = []
    #     images_labels = []
    #     images_dict = {'num_channels': 0, 'unique_labels': None}
    #
    #     num_images = 0
    #     for slice_id, files in self.slices_dict.items():
    #         if files['mri_image'] != '':
    #             num_images += 1
    #             mri_image = Image.open(files['mri_image'])
    #             if 'L' not in mri_image.mode:
    #                 mri_image = mri_image.convert('L')
    #
    #             images_dict['num_channels'] = mri_image.size if len(mri_image.size) > 2 else 1
    #             images_dict['image_shape'] = mri_image.size
    #
    #             if nparray:
    #                 images.append(np.array(mri_image))
    #             else:
    #                 images.append(mri_image)
    #
    #
    #         if files['mri_labels'] != '':
    #             mri_labels = Image.open(files['mri_labels'])
    #             if 'L' not in mri_labels.mode:
    #                 mri_labels = mri_labels.convert('L')
    #
    #             if images_dict['unique_labels'] is None:
    #                 images_dict['unique_labels'] = np.unique(mri_labels)
    #             else:
    #                 images_dict['unique_labels'] = list(set().union(np.unique(mri_labels), images_dict['unique_labels']))
    #
    #             if nparray:
    #                 images_labels.append(np.asarray(mri_labels))
    #             else:
    #                 images_labels.append(mri_labels)
    #
    #     if nparray:
    #         images = np.asarray(images)
    #         images_labels = np.asarray(images_labels)
    #
    #     images_dict['num_images'] = num_images
    #     return images, images_labels, images_dict
    #
    #
    # def load_landmarks(self):
    #     landmarks_dict = OrderedDict({slice_id: {'histo': {'X': [], 'Y': []}, 'mri': {'X': [], 'Y': []}}
    #                                   for slice_id in self.slices_dict.keys() })
    #
    #     for slice_id, files in self.slices_dict.items():
    #         if files['landmarks_file'] != '':
    #             with open(files['landmarks_file'], 'r') as readFile:
    #                 reader = csv.reader(readFile, delimiter=' ')
    #                 for row in reader:
    #                     landmarks_dict[slice_id]['histo']['X'].append(float(row[2]))
    #                     landmarks_dict[slice_id]['histo']['Y'].append(float(row[3]))
    #                     landmarks_dict[slice_id]['mri']['X'].append(float(row[0]))
    #                     landmarks_dict[slice_id]['mri']['Y'].append(float(row[1]))
    #
    #     return landmarks_dict
