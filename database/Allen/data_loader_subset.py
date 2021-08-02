import csv
from os.path import join, exists

import numpy as np
from PIL import Image


class Subject():
    def __init__(self, id, data_path, mri_dir, histo_dir, landmarks_dir, id_mm = None):

        self.data_path = data_path
        self._id = id
        self._id_mm = id_mm if id_mm is not None else id


        self.image_mri = join(mri_dir, 'images', 'slice_' + str(id) + '.png')
        self.image_histo = join(histo_dir, 'images', 'slice_' + str(id) + '.png')
        self.image_histo_affine = join(histo_dir, 'images_affine', 'slice_' + str(id) + '.png')
        self.labels_mri = join(mri_dir, 'labels', 'slice_' + str(id) + '_new.png') if exists(join(mri_dir, 'labels', 'slice_' + str(id) + '_new.png')) else join(mri_dir, 'labels', 'slice_' + str(id) + '.png')
        self.labels_histo = join(histo_dir, 'labels', 'slice_' + str(id) + '.png')
        self.labels_histo_affine = join(histo_dir, 'labels_affine', 'slice_' + str(id) + '_new.png') if exists(join(histo_dir, 'labels_affine', 'slice_' + str(id) + '_new.png')) else join(histo_dir, 'labels_affine', 'slice_' + str(id) + '.png')
        self.landmarks = join(data_path, 'landmarks','slice_' + str(id) + '.txt')
        self.landmarks_affine = join(data_path, 'landmarks_affine', 'slice_' + str(id) + '.txt')
        self.affine_histo2mri = join(data_path, 'affine_histo2mri','slice_' + str(id) + '.aff')


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

    def load_mri(self,*args, **kwargs):
        data = Image.open(self.image_mri)
        data = np.array(data)

        return data

    def load_histo(self,*args, **kwargs):
        data = Image.open(self.image_histo)
        data = np.array(data)
        # mask = self.load_histo_mask()
        # data = (1-data)*mask
        return data

    def load_histo_affine(self,*args, **kwargs):
        data = Image.open(self.image_histo_affine)
        data = np.array(data)
        # mask = self.load_histo_mask_affine()
        # data = (1 - data) * mask
        return data

    def load_mri_mask(self, *args, **kwargs):
        data = Image.open(self.labels_mri)
        data = np.array(data)

        return (np.sum(data, axis=-1) > 0).astype('float')

    def load_histo_mask(self, *args, **kwargs):

        data = Image.open(self.labels_histo)
        data = np.array(data)

        return (np.sum(data, axis=-1) > 0).astype('float')

    def load_histo_mask_affine(self, *args, **kwargs):
        data = Image.open(self.labels_histo_affine)
        data = np.array(data)

        return (np.sum(data, axis=-1) > 0).astype('float')

    def load_mri_labels(self, *args, **kwargs):
        data = Image.open(self.labels_mri)
        data = np.array(data)

        new_data = np.zeros(data.shape[:-1])
        new_data[(data[:,:,0] > 0) & (data[:,:,1] == 0) & (data[:,:,2] == 0)] = 1
        new_data[(data[:,:,0] == 0) & (data[:,:,1] > 0) & (data[:,:,2] == 0)] = 2
        new_data[(data[:,:,0] == 0) & (data[:,:,1] == 0) & (data[:,:,2] > 0)] = 3
        new_data[(data[:,:,0] > 0) & (data[:,:,1] > 0) & (data[:,:,2] > 0)] = 4
        # new_data = data[:,:,0]/255*1 + data[:,:,1]/255*2 + data[:,:,2]/255*3

        return new_data

    def load_histo_labels(self, *args, **kwargs):
        # data = Image.open(join(self.subject.dir_nissl, 'labels_orig', self.slice_name + '.png'))
        # data = np.array(data)

        data = Image.open(self.labels_histo)
        data = np.array(data)

        new_data = np.zeros(data.shape[:-1])
        print((data[:, :, 0] > 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0))
        new_data[(data[:, :, 0] > 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)] = 1
        new_data[(data[:, :, 0] == 0) & (data[:, :, 1] > 0) & (data[:, :, 2] == 0)] = 2
        new_data[(data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] > 0)] = 3
        new_data[(data[:, :, 0] > 0) & (data[:, :, 1] > 0) & (data[:, :, 2] > 0)] = 4
        #new_data = data[:,:,0]/255*1 + data[:,:,1]/255*2 + data[:,:,2]/255*3


        return new_data

    def load_histo_labels_affine(self, *args, **kwargs):
        # data = Image.open(join(self.subject.dir_nissl, 'labels_orig', self.slice_name + '.png'))
        # data = np.array(data)

        data = Image.open(self.labels_histo_affine)
        data = np.array(data)

        new_data = np.zeros(data.shape[:-1])
        new_data[(data[..., 0] > 0) & (data[..., 1] == 0) & (data[..., 2] == 0)] = 1
        new_data[(data[..., 0] == 0) & (data[..., 1] > 0) & (data[..., 2] == 0)] = 2
        new_data[(data[..., 0] == 0) & (data[..., 1] == 0) & (data[..., 2] > 0)] = 3
        new_data[(data[..., 0] > 0) & (data[..., 1] > 0) & (data[..., 2] > 0)] = 4
        #new_data = data[:,:,0]/255*1 + data[:,:,1]/255*2 + data[:,:,2]/255*3

        return new_data

    def load_mri_landmarks(self, *args, **kwargs):

        landmarks_list = []
        with open(self.landmarks, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_list.append((float(row[0]),float(row[1])))


        return landmarks_list

    def load_mri_landmarks_affine(self, *args, **kwargs):

        landmarks_list = []
        with open(self.landmarks_affine, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_list.append((float(row[0]),float(row[1])))


        return landmarks_list

    def load_histo_landmarks(self, *args, **kwargs):

        landmarks_list = []
        with open(self.landmarks, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_list.append((float(row[2]), float(row[3])))

        return landmarks_list

    def load_histo_landmarks_affine(self, *args, **kwargs):
        landmarks_list = []
        with open(self.landmarks_affine, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=' ')
            for row in reader:
                landmarks_list.append((float(row[2]),float(row[3])))

        return landmarks_list

    def load_affine_histo2mri(self, full=False, *args, **kwargs):

        with open(self.affine_histo2mri, 'r') as csvfile:
            if full:
                affine_matrix = np.zeros((4,4))
                csvreader = csv.reader(csvfile, delimiter=' ')
                for it_row, row in enumerate(csvreader):
                    affine_matrix[it_row, 0] = float(row[0])
                    affine_matrix[it_row, 1] = float(row[1])
                    affine_matrix[it_row, 2] = float(row[2])
                    affine_matrix[it_row, 3] = float(row[3])
            else:
                affine_matrix = np.zeros((2, 3))

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


class DataLoader(object):

    def __init__(self, database_config, **kwargs):
        self.database_config = database_config
        self.data_path = database_config['BASE_DIR']
        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, rid_list = None):
        self.n_dims = 2
        self.n_channels = 1

        self.subject_list = []
        self.rid_list = []
        self.subject_dict = {}
        with open(join(self.database_config['BASE_DIR'], self.database_config['DATA_FILE']), 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if rid_list is not None:
                    if row['SLICE_ID'] not in rid_list:
                        continue

                subject = Subject(row['SLICE_ID'],
                                  self.database_config['BASE_DIR'],
                                  self.database_config['MRI_DIR'],
                                  self.database_config['HISTOLOGY_DIR'],
                                  self.database_config['LANDMARKS_DIR'],
                                  )

                self.rid_list.append(row['SLICE_ID'])
                self.subject_list.append(subject)
                self.subject_dict[row['SLICE_ID']] = subject

    @property
    def image_shape(self):
        ishape = self.subject_list[0].image_shape
        for sbj in self.subject_list[1:]:
            sshape = sbj.image_shape
            ishape = tuple([max(a,b) for a,b in zip(ishape, sshape)])
        return ishape

    def __len__(self):
        return len(self.subject_list)
