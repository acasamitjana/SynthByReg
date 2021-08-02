import pdb
import subprocess
from os.path import join

import numpy as np
import torch
import nibabel as nib
from PIL import Image

from src.utils.image_utils import bilinear_interpolate
from src.utils import image_transforms as tf

NIFTY_REG_DIR = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'

def evaluate_rot(data_dict, model_dict, device, parameter_dict=None, init_image_shape=None, labels_flag=True,
                 landmarks_flag=True):

    init_mse = []
    reg_mse = []
    seg_M = 0
    seg_H = 0
    rseg_H = 0
    with torch.no_grad():
        H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
        M_m = data_dict['x_ref_mask'].to(device)
        M_s, H_s = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        img = H

        # Network
        _, H_f, _ = model_dict['R_M'](H, M)
        rH_s = model_dict['R_M'].predict(H_s, H_f, svf=False)*M_m
        H_s = H_s * M_m
        if labels_flag:
            seg_M = np.argmax(M_s[0, :].cpu().detach().numpy(), axis=0)
            seg_H = np.argmax(H_s[0, :].cpu().detach().numpy(), axis=0)
            rseg_H = np.argmax(rH_s[0, :].cpu().detach().numpy(), axis=0)

        if landmarks_flag:
            ref_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_ref_landmarks']]
            flo_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_flo_landmarks']]

            H = np.squeeze(H.cpu().detach().numpy())
            flow = np.squeeze(H_f.cpu().detach().numpy())
            if init_image_shape is not None and parameter_dict is not None:
                transform = tf.Compose(parameter_dict['TRANSFORM'])
                H, flowx, flowy = transform.inverse([H, flow[0], flow[1]], img_shape=[init_image_shape, init_image_shape, init_image_shape])
                flow = np.concatenate((flowx[np.newaxis], flowy[np.newaxis]), axis=0)

            num_landmarks = len(flo_landmarks)
            reg_landmarks = np.zeros((num_landmarks, 2))
            for it_l in range(num_landmarks):
                x, y = ref_landmarks[it_l]
                reg_landmarks[it_l, 1] = y + bilinear_interpolate(flow[0], x, y)
                reg_landmarks[it_l, 0] = x + bilinear_interpolate(flow[1], x, y)

            img = 0.8 * np.squeeze(H)[..., np.newaxis]
            img = np.concatenate((img, img, img), axis=-1)
            for it_l in range(num_landmarks):
                x1, y1 = ref_landmarks[it_l]
                img[int(y1):int(y1) + 3, int(x1):int(x1) + 3, 0] = 1 #red

                x2, y2 = reg_landmarks[it_l]
                img[int(y2):int(y2) + 3, int(x2):int(x2) + 3, 2] = 1 #blue

                x3, y3 = flo_landmarks[it_l]
                img[int(y3):int(y3) + 3, int(x3):int(x3) + 3] = 1 #white


                reg_mse.append(np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
                init_mse.append(np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))


    return img, seg_M, seg_H, rseg_H, [init_mse, reg_mse]


def evaluate_rbs(data_dict, model_dict, device, parameter_dict=None, init_image_shape=None, labels_flag=True,
                 landmarks_flag=True):

    init_mse = []
    reg_mse = []
    seg_M = 0
    seg_H = 0
    rseg_H = 0
    with torch.no_grad():
        H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
        M_m = data_dict['x_ref_mask'].to(device)
        M_s, H_s = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        img = H

        # Network
        M_g = model_dict['G_M'](H)
        _, H_f, _ = model_dict['R_M'](M_g, M)
        rH_s = model_dict['R_M'].predict(H_s, H_f, svf=False)*M_m
        H_s = H_s * M_m
        if labels_flag:
            seg_M = np.argmax(M_s[0, :].cpu().detach().numpy(), axis=0)
            seg_H = np.argmax(H_s[0, :].cpu().detach().numpy(), axis=0)
            rseg_H = np.argmax(rH_s[0, :].cpu().detach().numpy(), axis=0)

        if landmarks_flag:
            ref_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_ref_landmarks']]
            flo_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_flo_landmarks']]

            H = np.squeeze(H.cpu().detach().numpy())
            flow = np.squeeze(H_f.cpu().detach().numpy())
            if init_image_shape is not None and parameter_dict is not None:
                transform = tf.Compose(parameter_dict['TRANSFORM'])
                H, flowx, flowy = transform.inverse([H, flow[0], flow[1]], img_shape=[init_image_shape, init_image_shape, init_image_shape])
                flow = np.concatenate((flowx[np.newaxis], flowy[np.newaxis]), axis=0)

            num_landmarks = len(flo_landmarks)
            reg_landmarks = np.zeros((num_landmarks, 2))
            for it_l in range(num_landmarks):
                x, y = ref_landmarks[it_l]
                reg_landmarks[it_l, 1] = y + bilinear_interpolate(flow[0], x, y)
                reg_landmarks[it_l, 0] = x + bilinear_interpolate(flow[1], x, y)

            img = 0.8 * np.squeeze(H)[..., np.newaxis]
            img = np.concatenate((img, img, img), axis=-1)
            for it_l in range(num_landmarks):
                x1, y1 = ref_landmarks[it_l]
                img[int(y1):int(y1) + 3, int(x1):int(x1) + 3, 0] = 1 #red

                x2, y2 = reg_landmarks[it_l]
                img[int(y2):int(y2) + 3, int(x2):int(x2) + 3, 2] = 1 #blue

                x3, y3 = flo_landmarks[it_l]
                img[int(y3):int(y3) + 3, int(x3):int(x3) + 3] = 1 #white


                reg_mse.append(np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
                init_mse.append(np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))


    return img, seg_M, seg_H, rseg_H, [init_mse, reg_mse]


def evaluate_raw_register(data_dict, model, device, parameter_dict=None, init_image_shape=None, labels_flag=True,
                 landmarks_flag=True):

    init_mse = []
    reg_mse = []
    seg_M = 0
    seg_H = 0
    rseg_H = 0
    with torch.no_grad():

        H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
        M_m = data_dict['x_ref_mask'].to(device)
        M_s, H_s = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        img = H


        # Network
        _, H_f, _ = model(H, M)
        rH_s = model.predict(H_s, H_f, svf=False)
        H_s = H_s * M_m

        if labels_flag:
            seg_M = np.argmax(M_s[0, :].cpu().detach().numpy(), axis=0)
            seg_H = np.argmax(H_s[0, :].cpu().detach().numpy(), axis=0)
            rseg_H = np.argmax(rH_s[0, :].cpu().detach().numpy(), axis=0)

        if landmarks_flag:
            ref_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_ref_landmarks']]
            flo_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_flo_landmarks']]

            H = np.squeeze(H.cpu().detach().numpy())
            flow = np.squeeze(H_f.cpu().detach().numpy())
            if init_image_shape is not None and parameter_dict is not None:
                transform = tf.Compose(parameter_dict['TRANSFORM'])
                H, flowx, flowy = transform.inverse([H, flow[0], flow[1]],
                                                    img_shape=[init_image_shape, init_image_shape, init_image_shape])
                flow = np.concatenate((flowx[np.newaxis], flowy[np.newaxis]), axis=0)

            num_landmarks = len(flo_landmarks)
            reg_landmarks = np.zeros((num_landmarks, 2))
            for it_l in range(num_landmarks):
                x, y = ref_landmarks[it_l]
                reg_landmarks[it_l, 1] = y + bilinear_interpolate(flow[0], x, y)
                reg_landmarks[it_l, 0] = x + bilinear_interpolate(flow[1], x, y)


            img = 0.8 * np.squeeze(H)[..., np.newaxis]
            img = np.concatenate((img, img, img), axis=-1)
            for it_l in range(num_landmarks):
                x1, y1 = ref_landmarks[it_l]
                img[int(y1):int(y1) + 3, int(x1):int(x1) + 3, 0] = 1 # red

                x2, y2 = reg_landmarks[it_l]
                img[int(y2):int(y2) + 3, int(x2):int(x2) + 3, 2] = 1 #blue

                x3, y3 = flo_landmarks[it_l]
                img[int(y3):int(y3) + 3, int(x3):int(x3) + 3] = 1 #white

                reg_mse.append(np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
                init_mse.append(np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))

        # seg_M = np.argmax(M_s[0, :].cpu().detach().numpy(), axis=0)
        # seg_H = np.argmax(H_s[0, :].cpu().detach().numpy(), axis=0)
        # rseg_H = np.argmax(rH_s[0, :].cpu().detach().numpy(), axis=0)


    return img, seg_M, seg_H, rseg_H, [init_mse, reg_mse]


def evaluate_niftyreg_register(data_dict):


    # Filenames
    tempdir = '/tmp'
    refFile = join(tempdir, 'refFile.png')
    floFile = join(tempdir, 'floFile.png')
    refMaskFile = join(tempdir, 'refMaskFile.png')
    floMaskFile = join(tempdir, 'floMaskFile.png')

    outputFile = join(tempdir, 'outputFile.png')
    outputMaskFile = join(tempdir, 'outputMaskFile.png')
    nonlinearField = join(tempdir, 'nonlinearField.nii.gz')
    dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')


    H, M = data_dict['x_flo'], data_dict['x_ref']
    M_s, H_s = data_dict['x_ref_labels'], data_dict['x_flo_labels']
    M_m, H_m = data_dict['x_ref_mask'], data_dict['x_flo_mask']
    ref_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_ref_landmarks']]
    flo_landmarks = [[r.detach().numpy()[0] for r in ref] for ref in data_dict['x_flo_landmarks']]
    image_shape = M.shape


    # Save images
    img = Image.fromarray((255 * M).astype(np.uint8), mode='L')
    img.save(refFile)
    img = Image.fromarray((255 * H).astype(np.uint8), mode='L')
    img.save(floFile)
    img = Image.fromarray((255 * M_m).astype(np.uint8), mode='L')
    img.save(refMaskFile)
    img = Image.fromarray((255 * H_s).astype(np.uint8), mode='L')
    img.save(floMaskFile)

    # Network
    subprocess.call(
        [F3Dcmd, '-ref', refFile, '-flo', floFile, '--rmask', refMaskFile,'-res', outputFile, '-cpp', dummyFileNifti, '-sx', str(8),
         '-sy', str(8), '-ln', '4', '-lp', '3', '--nmi', '--rbn', '20', '--fbn', '20', '-pad', '0', '-vel', '-voff'],
        stdout=subprocess.DEVNULL)

    subprocess.call([TRANSFORMcmd, '-ref', refFile, '-flow', dummyFileNifti, nonlinearField], stdout=subprocess.DEVNULL)

    subprocess.call(
        [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', nonlinearField, '-res', outputMaskFile, '-inter',
         '0', '-voff'], stdout=subprocess.DEVNULL)

    rH_s = Image.open(outputMaskFile)
    rH_s = np.array(rH_s)

    YY, XX = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')
    proxy = nib.load(nonlinearField)
    proxyarray = np.transpose(np.squeeze(np.asarray(proxy.dataobj)), [2, 1, 0])
    proxyarray[np.isnan(proxyarray)] = 0
    velocity_field = np.zeros_like(proxyarray)
    velocity_field[0] = proxyarray[0] - XX
    velocity_field[1] = proxyarray[1] - YY

    nstep = 10
    flow_x = -velocity_field[0] / (2 ** nstep)
    flow_y = -velocity_field[1] / (2 ** nstep)
    for it_step in range(nstep):
        x = XX + flow_x
        y = YY + flow_y
        incx = bilinear_interpolate(flow_x, x, y)
        incy = bilinear_interpolate(flow_y, x, y)
        flow_x = flow_x + incx.reshape(image_shape)
        flow_y = flow_y + incy.reshape(image_shape)

    flow = np.concatenate((flow_x[np.newaxis], flow_y[np.newaxis]))

    # Quantitative
    num_landmarks = len(flo_landmarks)
    reg_landmarks = np.zeros((num_landmarks, 2))
    for it_l in range(num_landmarks):
        x, y = ref_landmarks[it_l]
        reg_landmarks[it_l, 1] = y + bilinear_interpolate(flow[0], x, y)
        reg_landmarks[it_l, 0] = x + bilinear_interpolate(flow[1], x, y)


    img = 0.8 * np.squeeze(H.cpu().detach().numpy())[..., np.newaxis]
    img = np.concatenate((img, img, img), axis=-1)
    init_mse = []
    reg_mse = []
    for it_l in range(num_landmarks):
        x1, y1 = ref_landmarks[it_l]
        img[int(y1):int(y1) + 3, int(x1):int(x1) + 3, 0] = 1 # red

        x2, y2 = reg_landmarks[it_l]
        img[int(y2):int(y2) + 3, int(x2):int(x2) + 3, 2] = 1 #blue

        x3, y3 = flo_landmarks[it_l]
        img[int(y3):int(y3) + 3, int(x3):int(x3) + 3] = 1 #white


        reg_mse.append(np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
        init_mse.append(np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))

    seg_M = np.argmax(M_s[0, :].cpu().detach().numpy(), axis=0)
    seg_H = np.argmax(H_s[0, :].cpu().detach().numpy(), axis=0)
    rseg_H = np.argmax(rH_s[0, :].cpu().detach().numpy(), axis=0)


    return img, seg_M, seg_H, rseg_H, [init_mse, reg_mse], flow

