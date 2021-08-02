# imports
from os.path import join
from datetime import date, datetime
from argparse import ArgumentParser
import subprocess

# third party imports
import numpy as np
import torch

# project imports
from src import losses, models, datasets
from src.utils.io import DebugWriter, ResultsWriter, create_results_dir, ExperimentWriter, worker_init_fn
from src.utils.tensor_utils import TensorDeformation
from src.callbacks import LRDecay
from src.training import Registration
from scripts.Registration.BigBrain import configFileBigBrain as configFile

####################################
######## GLOBAL  PARAMETERS ########
####################################
NIFTY_REG_DIR = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'


DEBUG_FLAG = False
debugWriter = DebugWriter(DEBUG_FLAG)
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

parameter_dict = configFile.CONFIG

attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False

bidir_flag = True#False if stain_flag != 'MRI' else True
mask_flag = True
kwargs_training = {'log_interval': parameter_dict['LOG_INTERVAL'], 'bidir_flag': bidir_flag, 'mask_flag': mask_flag}  # Number of steps
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
CREATE_MASKS = False
AFFINE_TF = True

from PIL import Image
from skimage import measure
from skimage.morphology import binary_opening, disk
from scipy.ndimage.morphology import binary_fill_holes
selem = disk(3)
selem_holes = disk(4)

for rid, subject in data_loader.subject_dict.items():
    # if int(rid) >= 16:
    #     continue
    print(rid)

    if CREATE_MASKS:
        H = subject.load_histo()
        M = H < 250
        M = binary_fill_holes(M, selem_holes)
        M = binary_opening(M, selem)
        all_blobs, num_blobs = measure.label(M, connectivity=1, return_num=True)
        all_blobs_count = np.bincount(all_blobs[M > 0])
        unique_blobs = np.unique(all_blobs[M > 0])
        for it_abc, abc in enumerate(all_blobs_count):
            if abc < 500 and np.max(all_blobs_count) > 500:
                M[all_blobs==it_abc] = 0

        img = Image.fromarray((255*M).astype(np.uint8), mode='L')
        img.save(subject.labels_histo)

    if AFFINE_TF:
        # subprocess.call([
        #     ALADINcmd, '-ref', subject.image_mri, '-flo', subject.image_histo, '-aff', subject.matrix_histo_affine,
        #     '-res', subject.image_histo_affine, '-rmask', subject.labels_mri, '-ln', '4', '-lp', '3', '-pad', '0',
        # ])
        #
        # subprocess.call(
        #     [REScmd, '-ref', subject.image_mri, '-flo', subject.labels_histo,
        #      '-trans', subject.matrix_histo_affine, '-res', subject.labels_histo_affine,
        #      '-inter', '0', '-voff'])
        pass



print('Done.')
