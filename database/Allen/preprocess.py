# imports
from os.path import join
from datetime import date, datetime
from argparse import ArgumentParser
import subprocess

# third party imports
import numpy as np
import torch
from PIL import Image

# project imports
from database.Allen import data_loader_subset as DL_Allen_subset
from database import databaseConfig

####################################
######## GLOBAL  PARAMETERS ########
####################################
NIFTY_REG_DIR = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'


MRI_PROCESSING = False
LABELS_PROCESSING = False
###################################
########### DATA LOADER ###########
###################################
if MRI_PROCESSING:
    data_loader = DL_Allen_subset.DataLoader(databaseConfig.ALLEN_subset)
    MASK_MRI = False
    COMPUTE_CROP_COORDINATES = False
    mx = 0
    Mx = 448
    my = 0
    My = 320
    for rid, subject in data_loader.subject_dict.items():
        # 1 Mask MRI
        if MASK_MRI:
            MM = subject.load_mri_mask()
            MI = subject.load_mri()
            MI[MM == 0] = 0
            img = Image.fromarray(MI, mode='L')
            img.save(subject.image_mri)

        # 2 Compute cropping coordinates
        if COMPUTE_CROP_COORDINATES:
            MM = subject.load_mri_mask()
            idx = np.where(MM > 0)
            mx = min(mx, np.min(idx[0]))
            Mx = max(Mx, np.max(idx[0]))
            my = min(my, np.min(idx[1]))
            My = max(My, np.max(idx[1]))

        # 3 Crop
        MI = subject.load_mri()
        MM = Image.open(subject.labels_mri)
        MM = np.array(MM)
        HI = subject.load_histo_affine()
        HM = Image.open(subject.labels_histo_affine)
        HM = np.array(HM)

        MMc = MM[mx:Mx, my:My]
        MIc = MI[mx:Mx, my:My]
        HMc = HM[mx:Mx, my:My]
        HIc = HI[mx:Mx, my:My]
        img = Image.fromarray(MIc, mode='L')
        img.save(subject.image_mri)
        img = Image.fromarray(MMc, mode='RGB')
        img.save(subject.labels_mri)
        img = Image.fromarray(HIc, mode='L')
        img.save(subject.image_histo_affine)
        img = Image.fromarray(HMc, mode='RGB')
        img.save(subject.labels_histo_affine)

    print('Done.')


#################
# WM Cerebellum #
#################
import pdb
if LABELS_PROCESSING:
    data_loader = DL_Allen_subset.DataLoader(databaseConfig.ALLEN_subset_cll)
    data_loader_all = DL_Allen_subset.DataLoader(databaseConfig.ALLEN_subset)

    for rid, subject in data_loader.subject_dict.items():
        subject_full = data_loader_all.subject_dict[rid.split('.')[0]]

        # 1. Histo Affine
        labels_cll = subject.load_histo_labels_affine()
        labels = subject_full.load_histo_labels_affine()

        new_labels = np.zeros(labels_cll.shape + (3,), dtype='uint8')
        new_labels[labels == 1, 0] = 255
        new_labels[labels == 2, 1] = 255
        new_labels[labels == 3, 2] = 255

        new_labels[labels_cll == 1, 0] = 255
        new_labels[labels_cll == 2, :] = 255

        img = Image.fromarray(new_labels, mode='RGB')
        img.save(subject_full.labels_histo_affine[:-4] + '_new.png')

        # 2. MRI
        labels_cll = subject.load_mri_labels()
        labels = subject_full.load_mri_labels()

        new_labels = np.zeros(labels_cll.shape + (3,), dtype='uint8')
        new_labels[labels == 1, 0] = 255
        new_labels[labels == 2, 1] = 255
        new_labels[labels == 3, 2] = 255

        new_labels[labels_cll == 1, 0] = 255
        new_labels[labels_cll == 2, :] = 255

        img = Image.fromarray(new_labels, mode='RGB')
        img.save(subject_full.labels_mri[:-4] + '_new.png')