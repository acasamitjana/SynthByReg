# imports
from os.path import join
from datetime import date, datetime
from argparse import ArgumentParser
import subprocess

# third party imports
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage.filters import laplace, gaussian_filter
from skimage.morphology import binary_opening, remove_small_holes, binary_closing
from skimage import measure

# project imports
from src.utils.io import DebugWriter, ResultsWriter, create_results_dir, ExperimentWriter
from scripts.Registration.L2R import configFileL2R as configFile
from src.utils.image_transforms import ScaleNormalization

####################################
######## GLOBAL  PARAMETERS ########
####################################
DEBUG_FLAG = False
debugWriter = DebugWriter(DEBUG_FLAG)
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

NIFTY_REG_DIR = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'

""" PARSE ARGUMENTS FROM CLI """
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--contrast', help='stain used', default='MRI', choices=['CT', 'MRI']),

arguments = arg_parser.parse_args()
stain_flag = arguments.contrast

if stain_flag == 'CT':
    parameter_dict = configFile.CONFIG_CT
elif stain_flag == 'MRI':
    parameter_dict = configFile.CONFIG_MRI
else:
    raise ValueError

create_results_dir(parameter_dict['RESULTS_DIR'])
attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False

kwargs_training = {'log_interval': parameter_dict['LOG_INTERVAL']}  # Number of steps
device = torch.device("cuda:0" if parameter_dict['USE_GPU'] else "cpu")

debugWriter = DebugWriter(DEBUG_FLAG)
resultsWriter = ResultsWriter(join(parameter_dict['RESULTS_DIR'], 'experiment_parameters.txt'), attach=attach)
experimentWriter = ExperimentWriter(join(parameter_dict['RESULTS_DIR'], 'experiment.txt'), attach=attach)


resultsWriter.write('Experiment parameters\n')
for key, value in parameter_dict.items():
    resultsWriter.write(key + ': ' + str(value))
    resultsWriter.write('\n')
resultsWriter.write('\n')

###################################
########### DATA LOADER ###########
###################################
resultsWriter.write('Loading dataset ...\n')
data_loader = parameter_dict['DB_CONFIG']['DATA_LOADER'].DataLoader(parameter_dict['DB_CONFIG'])
strel_op = np.zeros((7,7,7))
strel_op[:,3,3] = 1
strel_op[3,:,3] = 1
strel_op[3,3,:] = 1
strel_clo = np.zeros((9,9,9))
strel_clo[:,5,5] = 1
strel_clo[5,:,5] = 1
strel_clo[5,5,:] = 1
MRI_MASK = False
CT_MASK = False
RESAMPLE = False
AFFINE_TF = False
MRI = True
for rid, subject in data_loader.subject_dict.items():
    print(rid)

    # MRI mask
    if MRI_MASK:
        proxy = nib.load(subject.image_mri)
        data = np.asarray(proxy.dataobj)

        M = np.quantile(data, 0.99)
        m = np.quantile(data, 0.01)

        data = (data - m) / (M - m)

        if rid == '0014_tcia':
            th = 0.15
            area_th = 100000
        elif rid == '0010_tcia':
            th = 0.05
            area_th = 100000
            strel_clo[:, :, 5] = 1
        elif rid == '0006_tcia':
            th = 0.075
            area_th = 200000
            strel_clo[:, :, 5] = 1
        else:
            th = 0.05
            area_th = 1500

        mask = data > th
        mask = binary_opening(mask, strel_op)
        mask = binary_closing(mask, strel_clo)
        mask = remove_small_holes(mask, area_threshold=area_th)
        img = nib.Nifti1Image(mask.astype('uint8'), subject.vox2ras)
        nib.save(img, subject.mask_mri)

    # CT mask
    if CT_MASK:
        proxy = nib.load(subject.image_ct)
        data = np.asarray(proxy.dataobj)
        data = np.clip(data, -150, 70)
        img = nib.Nifti1Image(data, subject.vox2ras)
        nib.save(img, subject.image_ct)
        mask = data > -150
        mask = binary_opening(mask, strel_op)
        mask = remove_small_holes(mask, area_threshold=20)

        img = nib.Nifti1Image(mask.astype('uint8'), subject.vox2ras)
        nib.save(img, subject.mask_ct)

    if AFFINE_TF:
        subprocess.call([
            ALADINcmd, '-ref', subject.image_mri, '-flo', subject.image_ct, '-aff', subject.matrix_ct_affine,
            '-res', subject.image_ct_affine, '-rmask', subject.mask_mri, '-ln', '4', '-lp', '3', '-pad', '0',
            '-nac', '-fmask', subject.mask_ct
        ])

    if RESAMPLE:
        # subprocess.call(
        #     [REScmd, '-ref', subject.image_mri, '-flo', subject.image_ct,
        #      '-trans', subject.matrix_ct_affine, '-res', subject.image_ct_affine,
        #      '-inter', '2', '-voff'])
        #
        subprocess.call(
            [REScmd, '-ref', subject.image_mri, '-flo', subject.mask_ct,
             '-trans', subject.matrix_ct_affine, '-res', subject.mask_ct_affine,
             '-inter', '0', '-voff'])

        # subprocess.call(
        #     [REScmd, '-ref', subject.image_mri, '-flo', subject.seg_ct,
        #      '-trans', subject.matrix_ct_affine, '-res', subject.seg_ct_affine,
        #      '-inter', '0', '-voff'])

    if MRI:
        proxy = nib.load(subject.image_mri)
        data = np.asarray(proxy.dataobj)
        vox2ras0 = proxy.affine

        proxy = nib.load(subject.mask_mri)
        mask = np.asarray(proxy.dataobj)

        norm = ScaleNormalization(range=[0, 1], quantile=True, contrast=[0.9, 0.2])
        data = norm(data, mask)

        img = nib.Nifti1Image(data, vox2ras0)
        nib.save(img, subject.image_mri)

print('Done.')
