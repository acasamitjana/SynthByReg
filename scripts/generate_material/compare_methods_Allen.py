import functools
from os.path import join, exists
from os import makedirs
from datetime import date, datetime
import pdb
import itertools

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, font_manager as fm
from scipy.stats import ttest_rel, wilcoxon

from src import models, datasets, metrics, evaluate as evaluate_class
from src.utils.image_utils import one_hot_encoding
from src.utils.visualization import slices, plot_results
from scripts.InfoNCE.Allen_labels import configFileAllen as configFile
from src.utils.io import ExperimentWriter
plt.switch_backend('Agg')

rid_list = None#['055', '035', '085']
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

####################################
############ PARAMETERS ############
####################################
RESULTS_DIR = '/home/acasamitjana/Results/RegSyn/Allen_labels/results'
model_ckp_list = {
    'NMI': '/home/acasamitjana/Results/RegSyn/Allen_labels/Registration/MRI_HISTO_NMI/D2_R1_S1/sigmoid_bidir_2neigh_noDA/checkpoints/model_checkpoint.FI.pth',
    'cGAN': '/home/acasamitjana/Results/RegSyn/Allen_labels/CycleGAN/2_NCE_SIGMOID/HISTO_R1_RS1_C1_G1/checkpoints/model_checkpoint.FI.pth',
    'RoT': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/RoT/HISTO__RL1_RS1_G1/checkpoints/model_checkpoint.FI.pth',
    'SbR-N': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.0_G0.0_T0.0/checkpoints/model_checkpoint.FI.pth',
    'SbR': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.02_G0.0_T0.05/checkpoints/model_checkpoint.FI.pth',
    ## 'SbR-R': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/REFINE/HISTO_reg__RL1.0_RN0.0_RS1.0_N0.02_G0.0_T0.05_E0/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.02_G0.0_T0.05/checkpoints/model_checkpoint.LAST.pth',
    ## 'SbR-R2': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/REFINE/HISTO_reg__RL1.0_RN0.0_RS1.0_N0.02_G0.0_T0.05_E0_onlyR/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.02_G0.0_T0.05/checkpoints/model_checkpoint.LAST.pth',
    'SbR-G': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.02_G1_T0.05/checkpoints/model_checkpoint.LAST.pth',
    'SbR-R': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/REFINE/HISTO_reg__RL1.0_RN0.0_RS1.0_N0.02_G0.0_T0.05_E0_higherLR/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.02_G0.0_T0.05/checkpoints/model_checkpoint.LAST.pth',
    # 'SbR2': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/HISTO_reg__RL5.0_RN0.0_RS0.0_N0.1_G0.0_T0.05/checkpoints/model_checkpoint.LAST.pth',
    # 'SbR3': '/home/acasamitjana/Results/RegSyn/Allen_labels/InfoNCE/DEF/2_NCE_SIGMOID/HISTO_reg__RL1.0_RN0.0_RS0.0_N0.02_G0.0_T0.05/checkpoints/model_checkpoint.LAST.pth',
    'NMIw': '/home/acasamitjana/Results/RegSyn/Allen_labels/Registration/MRI_HISTO_WEAKLY_NMI/D2_R1_RL1_S1/sigmoid_bidir_2neigh_noDA/checkpoints/model_checkpoint.FI.pth',
}

parameter_dict = configFile.CONFIG

kwargs_testing = {}
use_gpu = torch.cuda.is_available() and parameter_dict['USE_GPU']
device = torch.device("cuda:0" if use_gpu else "cpu")
###################################
########### DATA LOADER ###########
###################################
input_channels_mri = 1
input_channels_histo = 1
nonlinear_field_size = [9, 9]

db_loader = parameter_dict['DB_CONFIG']['DATA_LOADER']
data_loader = db_loader.DataLoader(parameter_dict['DB_CONFIG'], rid_list=rid_list)

dataset_test = datasets.InterModalRegistrationDataset2D(
    data_loader,
    rotation_params=parameter_dict['ROTATION'],
    nonlinear_params=parameter_dict['NONLINEAR'],
    ref_modality='MRI',
    flo_modality='HISTO',
    tf_params=parameter_dict['TRANSFORM'],
    da_params=parameter_dict['DATA_AUGMENTATION'],
    norm_params=parameter_dict['NORMALIZATION'],
    num_classes=parameter_dict['N_CLASSES_SEGMENTATION'],
    mask_dilation=np.ones((7, 7)),
    train=True,
    landmarks=True
)


generator_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)

image_size_dict = {'M': dataset_test.image_shape, 'H': dataset_test.image_shape, 'latent': parameter_dict['N_CLASSES_NETWORK']}

#################################
############# MODEL #############
#################################
image_shape = dataset_test.image_shape

# Generator Network
G_M = models.ResnetGenerator(
    input_nc=input_channels_mri,
    output_nc=input_channels_histo,
    ngf=32,
    n_blocks=6,
    norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
    n_downsampling=3,
    device=device,
    tanh=True if 'TANH' in parameter_dict['PARENT_DIRECTORY'] else False
)

registration_M = models.VxmDense(
    nb_unet_features=[parameter_dict['ENC_NF_REG'], parameter_dict['DEC_NF_REG']],
    inshape=image_shape,
    int_steps=7,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)

G_M = G_M.to(device)
registration_M = registration_M.to(device)


model_dict = {'G_M': G_M, 'R_M': registration_M}

##################
#   Prediction   #
##################
FI_MSE = {'RegType': [], 'Error': []}
FI_D0 = {'RegType': [], 'Error': []}
FI_D1 = {'RegType': [], 'Error': []}
FI_D2 = {'RegType': [], 'Error': []}
FI_D3 = {'RegType': [], 'Error': []}
FI_D4 = {'RegType': [], 'Error': []}

for reg_type, weightsfile in model_ckp_list.items():
    checkpoint = torch.load(weightsfile, map_location=device)
    if 'SbR' in reg_type or 'GAN' in reg_type or 'RoT' in reg_type:
        for model_key, model in model_dict.items():
            model.load_state_dict(checkpoint['state_dict_' + model_key])
            model.eval()

    else:
        model_dict['R_M'].load_state_dict(checkpoint['state_dict'])
        model_dict['R_M'].eval()

    modality_list = ['M', 'H']
    results_dir = join(RESULTS_DIR, reg_type)
    for it_image in range(len(dataset_test)):
        if not exists(join(results_dir, str(data_loader.rid_list[it_image]))):
            makedirs(join(results_dir, str(data_loader.rid_list[it_image])))

    total_init_mse = []
    total_fi_mse = []

    total_init_dice_0 = []
    total_init_dice_1 = []
    total_init_dice_2 = []
    total_init_dice_3 = []
    total_init_dice_4 = []

    total_fi_dice_0 = []
    total_fi_dice_1 = []
    total_fi_dice_2 = []
    total_fi_dice_3 = []
    total_fi_dice_4 = []

    num_landmarks = []
    for batch_idx, data_dict in enumerate(generator_test):
        num_landmarks.append(len(data_dict['x_ref_landmarks']))

        print(str(batch_idx) + '/' + str(len(generator_test)) + '. Slice: ' + str(data_dict['rid'][0]))
        results_sbj_dir = join(results_dir, str(data_dict['rid'][0]))

        if 'SbR' in reg_type or 'GAN' in reg_type:
            output_results = evaluate_class.evaluate_rbs(data_dict, model_dict, device)

        elif 'RoT' in reg_type:
            output_results = evaluate_class.evaluate_rot(data_dict, model_dict, device)

        else:
            output_results = evaluate_class.evaluate_raw_register(data_dict, model_dict['R_M'], device)

        H, seg_M, seg_H, rseg_H, landmarks = output_results

        init_mse, reg_mse = landmarks
        total_init_mse.append(np.mean(init_mse))
        total_fi_mse.append(np.mean(reg_mse))

        img = Image.fromarray((255 * H).astype(np.uint8), mode='RGB')
        img.save(join(results_sbj_dir, 'H_landmarks.png'))

        seg_M = one_hot_encoding(seg_M, parameter_dict['N_CLASSES_SEGMENTATION'])
        seg_H = one_hot_encoding(seg_H, parameter_dict['N_CLASSES_SEGMENTATION'])
        rseg_H = one_hot_encoding(rseg_H, parameter_dict['N_CLASSES_SEGMENTATION'])

        dice_0 = metrics.dice_coefficient(seg_M[0], seg_H[0])
        total_init_dice_0.append(dice_0)
        dice_0 = metrics.dice_coefficient(seg_M[0], rseg_H[0])
        total_fi_dice_0.append(dice_0)

        dice_1 = metrics.dice_coefficient(seg_M[1], seg_H[1])
        dice_1 = 0 if np.isnan(dice_1) else dice_1
        total_init_dice_1.append(dice_1)
        dice_1 = metrics.dice_coefficient(seg_M[1], rseg_H[1])
        dice_1 = 0 if np.isnan(dice_1) else dice_1
        total_fi_dice_1.append(dice_1)

        dice_2 = metrics.dice_coefficient(seg_M[2], seg_H[2])
        total_init_dice_2.append(dice_2)
        dice_2 = metrics.dice_coefficient(seg_M[2], rseg_H[2])
        total_fi_dice_2.append(dice_2)

        dice_3 = metrics.dice_coefficient(seg_M[3], seg_H[3])
        total_init_dice_3.append(dice_3)
        dice_3 = metrics.dice_coefficient(seg_M[3], rseg_H[3])
        total_fi_dice_3.append(dice_3)

        dice_4 = metrics.dice_coefficient(seg_M[4], seg_H[4])
        dice_4 = 0 if np.isnan(dice_4) or np.sum(seg_M[4]) == 0 or np.sum(seg_H[4]) == 0 else dice_4
        total_init_dice_4.append(dice_4)
        dice_4 = metrics.dice_coefficient(seg_M[4], rseg_H[4])
        dice_4 = 0 if np.isnan(dice_4) or np.sum(seg_M[4]) == 0 or np.sum(rseg_H[4]) == 0 else dice_4
        total_fi_dice_4.append(dice_4)

        seg_M[1] += seg_M[-1]
        seg_M[2] += seg_M[-1]
        seg_M[3] += seg_M[-1]
        seg_M = seg_M[:-1]
        img = Image.fromarray((255 * np.transpose(seg_M[1:], axes=[1, 2, 0])).astype(np.uint8), mode='RGB')
        img.save(join(results_sbj_dir, 'M_seg.png'))

        seg_H[1] += seg_H[-1]
        seg_H[2] += seg_H[-1]
        seg_H[3] += seg_H[-1]
        seg_H = seg_H[:-1]
        img = Image.fromarray((255 * np.transpose(seg_H[1:], axes=[1, 2, 0])).astype(np.uint8), mode='RGB')
        img.save(join(results_sbj_dir, 'H_seg.png'))

        rseg_H[1] += rseg_H[-1]
        rseg_H[2] += rseg_H[-1]
        rseg_H[3] += rseg_H[-1]
        rseg_H = rseg_H[:-1]
        img = Image.fromarray((255 * np.transpose(rseg_H[1:], axes=[1, 2, 0])).astype(np.uint8), mode='RGB')
        img.save(join(results_sbj_dir, 'H_reg_seg.png'))

    plt.figure()
    plt.plot(total_init_mse, 'r', marker = '*', label='Initial')
    plt.plot(total_fi_mse, 'b', marker='*', label='Fi')
    plt.grid()
    plt.legend()
    plt.title('MSE Init: ' + str(np.round(np.mean(total_init_mse),2)) + '. MSE Fi: ' + str(np.round(np.mean(total_fi_mse),2)))
    plt.savefig(join(results_dir, 'landmark_results.png'))
    plt.close()

    plt.figure()
    plt.plot(total_init_dice_0, 'r', marker = '*', label='Initial')
    plt.plot(total_fi_dice_0, 'b', marker='*', label='Fi')
    plt.grid()
    plt.legend()
    plt.title('Dice Bkg Init: ' + str(np.round(np.mean(total_init_dice_0),2)) + '. Dice Bkg Fi: ' + str(np.round(np.mean(total_fi_dice_0),2)))
    plt.savefig(join(results_dir, 'dice_0.png'))
    plt.close()

    plt.figure()
    plt.plot(total_init_dice_1, 'r', marker = '*', label='Initial')
    plt.plot(total_fi_dice_1, 'b', marker='*', label='Fi')
    plt.grid()
    plt.legend()
    plt.title('GMc Init: ' + str(np.round(np.mean([m for m in total_init_dice_1 if m>0]),2)) + '. Dice GMc Fi: ' + str(np.round(np.mean([m for m in total_fi_dice_1 if m>0]),2)))
    plt.savefig(join(results_dir, 'dice_1.png'))
    plt.close()

    plt.figure()
    plt.plot(total_init_dice_2, 'r', marker = '*', label='Initial')
    plt.plot(total_fi_dice_2, 'b', marker='*', label='Fi')
    plt.grid()
    plt.legend()
    plt.title('Dice WM Init: ' + str(np.round(np.mean(total_init_dice_2),2)) + '. Dice WM Fi: ' + str(np.round(np.mean(total_fi_dice_2),2)))
    plt.savefig(join(results_dir, 'dice_2.png'))
    plt.close()

    plt.figure()
    plt.plot(total_init_dice_3, 'r', marker = '*', label='Initial')
    plt.plot(total_fi_dice_3, 'b', marker='*', label='Fi')
    plt.grid()
    plt.legend()
    plt.title('Dice GM Init: ' + str(np.round(np.mean(total_init_dice_3),2)) + '. Dice GM Fi: ' + str(np.round(np.mean(total_fi_dice_3),2)))
    plt.savefig(join(results_dir, 'dice_3.png'))
    plt.close()


    plt.figure()
    plt.plot(total_init_dice_4, 'r', marker = '*', label='Initial')
    plt.plot(total_fi_dice_4, 'b', marker='*', label='Fi')
    plt.grid()
    plt.legend()
    plt.title('WMc Init: ' + str(np.round(np.mean([m for m in total_init_dice_4 if m>0]),2)) + '. Dice WMc Fi: ' + str(np.round(np.mean([m for m in total_fi_dice_4 if m>0]),2)))
    plt.savefig(join(results_dir, 'dice_4.png'))
    plt.close()

    if not FI_MSE['RegType']:
        FI_MSE['RegType'] += ['Linear']*len(total_init_mse)
        FI_MSE['Error'] += total_init_mse

    if not FI_D0['RegType']:
        FI_D0['RegType'] += ['Linear'] * len(total_init_mse)
        FI_D0['Error'] += total_init_dice_0

    if not FI_D1['RegType']:
        FI_D1['Error'] += [t for t in total_init_dice_1 if t > 0]
        FI_D1['RegType'] += ['Linear'] * len(FI_D1['Error'])

    if not FI_D2['RegType']:
        FI_D2['RegType'] += ['Linear'] * len(total_init_dice_2)
        FI_D2['Error'] += total_init_dice_2

    if not FI_D3['RegType']:
        FI_D3['RegType'] += ['Linear'] * len(total_init_dice_3)
        FI_D3['Error'] += total_init_dice_3

    if not FI_D4['RegType']:
        FI_D4['Error'] += [t for t in total_init_dice_4 if t > 0]
        FI_D4['RegType'] += ['Linear'] * len(FI_D4['Error'])

    FI_MSE['RegType'] += [reg_type] * len(total_fi_mse)
    FI_MSE['Error'] += total_fi_mse

    FI_D0['RegType'] += [reg_type] * len(total_fi_dice_0)
    FI_D0['Error'] += total_fi_dice_0

    if np.sum(total_fi_dice_1) > 0:
        FI_D1['Error'] += [t2 for t1, t2 in zip(total_init_dice_1, total_fi_dice_1) if t1 > 0]
        FI_D1['RegType'] += [reg_type] * len([t for t in total_init_dice_1 if t > 0])


    FI_D2['RegType'] += [reg_type] * len(total_fi_dice_2)
    FI_D2['Error'] += total_fi_dice_2

    FI_D3['RegType'] += [reg_type] * len(total_fi_dice_3)
    FI_D3['Error'] += total_fi_dice_3

    if np.sum(total_fi_dice_4) > 0:
        FI_D4['Error'] += [t2 for t1, t2 in zip(total_init_dice_4, total_fi_dice_4) if t1 > 0]
        FI_D4['RegType'] += [reg_type] * len([t for t in total_init_dice_4 if t > 0])


fpath = '/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf'#'/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf'
prop_bold = fm.FontProperties(fname=fpath)
fpath = '/usr/share/fonts/truetype/msttcorefonts/arial.ttf'
prop = fm.FontProperties(fname=fpath)
prop_legend = fm.FontProperties(fname=fpath, size=16)

data_frame = pd.DataFrame(FI_MSE)
plt.figure()
x = sns.boxplot(x="RegType", y="Error", data=data_frame, linewidth=2.5)
x.grid()
x.set_axisbelow(True)
x.locator_params(axis='y', tight=True, nbins=6)
plt.axes(x)
handles, labels = x.get_legend_handles_labels()
x.legend(handles, labels, loc=2, ncol=2, prop=prop_legend)#, bbox_to_anchor=(0.5, 1.05))
x.set_title('Landmark error',fontproperties=prop, fontsize=20)# y=1.0, pad=42, )
# handles, labels = x.get_legend_handles_labels()
# l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.ylabel('RMSE (pixels)', fontproperties=prop_bold, fontsize=18)
plt.yticks(rotation=90, fontproperties=prop, fontsize=16)
plt.xticks(fontproperties=prop, fontsize=16, rotation=20)
plt.xlabel('')
# plt.title(title_plot, fontproperties=prop, fontsize=20)
plt.legend([], [], frameon=False)

plt.savefig(join(RESULTS_DIR, 'MSE.png'))
plt.close()

dice_dict = {
    0: 'Background',
    1: 'GM cerebellum',
    2: 'WM cerebrum',
    3: 'GM cerebrum',
    4: 'WM cerebellum'
}

DICE = {
    0: FI_D0,
    1: FI_D1,
    2: FI_D2,
    3: FI_D3,
    4: FI_D4,
}
for it_d, string in dice_dict.items():
    data_frame = pd.DataFrame(DICE[it_d])
    plt.figure()
    x = sns.boxplot(x="RegType", y="Error", data=data_frame, linewidth=2.5)
    x.grid()
    x.set_axisbelow(True)
    x.locator_params(axis='y', tight=True, nbins=6)
    plt.axes(x)
    handles, labels = x.get_legend_handles_labels()
    x.legend(handles, labels, loc=2, ncol=2, prop=prop_legend)  # , bbox_to_anchor=(0.5, 1.05))
    x.set_title(string, fontproperties=prop, fontsize=20)  # y=1.0, pad=42, )
    # handles, labels = x.get_legend_handles_labels()
    # l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.ylabel('Dice coefficient', fontproperties=prop_bold, fontsize=18)
    plt.yticks(rotation=90, fontproperties=prop, fontsize=16)
    plt.xticks(fontproperties=prop, fontsize=16, rotation=20)
    plt.xlabel('')
    # plt.title(title_plot, fontproperties=prop, fontsize=20)
    plt.legend([], [], frameon=False)

    plt.savefig(join(RESULTS_DIR, 'Dice_' + str(it_d) + '.png'))
    plt.close()


reg_type_list = ['Linear'] + list(model_ckp_list.keys())
data_frame = pd.DataFrame(FI_MSE)

expWriter = ExperimentWriter(join(RESULTS_DIR, 'numerical_experiment.txt'), attach=False)
expWriter.write('Root-MSE (in pixels)\n')
for reg_type_1 in reg_type_list:
    idx_1 = data_frame['RegType'] == reg_type_1
    data_1 = data_frame['Error'][idx_1]

    expWriter.write(reg_type_1 + ' -->')
    expWriter.write(' Mean: ' + str(np.round(np.mean(data_1), 4)))
    expWriter.write(' Median: ' + str(np.round(np.median(data_1), 4)))
    expWriter.write(' Variance: ' + str(np.var(data_1)))
    expWriter.write('\n')

expWriter.write('Statistical significane:\n')
for reg_type_1, reg_type_2 in itertools.combinations(reg_type_list,2):

    idx_1 = data_frame['RegType'] == reg_type_1
    idx_2 = data_frame['RegType'] == reg_type_2
    data_1 = data_frame['Error'][idx_1]
    data_2 = data_frame['Error'][idx_2]

    t, p = wilcoxon(x=data_1, y=data_2, alternative="greater")
    expWriter.write(reg_type_1 + ' and ' + reg_type_2 + ': t_G=' + str(t) + ', p_G=' + str(p))
    t, p = wilcoxon(x=data_1, y=data_2, alternative="less")
    expWriter.write(reg_type_1 + ' and ' + reg_type_2 + ': t_L=' + str(t) + ', p_L=' + str(p))
    expWriter.write('\n')

print('Done.')