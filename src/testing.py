import numpy as np
import torch

from src import models

def predict(generator, model_dict, prediction_size_dict, device, da_model = None):
    num_elements = len(generator.dataset)
    num_batches = len(generator)
    batch_size = generator.batch_size
    ndims = len(prediction_size_dict['M'])
    data_M = np.zeros((num_elements,) + prediction_size_dict['M'])
    data_H = np.zeros((num_elements,) + prediction_size_dict['H'])
    mask_M = np.zeros((num_elements,) + prediction_size_dict['M'])
    mask_H = np.zeros((num_elements,) + prediction_size_dict['H'])
    labels_M = np.zeros((num_elements,) + prediction_size_dict['M'])

    gen_M = np.zeros((num_elements,) + prediction_size_dict['M'])

    reg_data_H = np.zeros((num_elements,) + prediction_size_dict['H'])
    reg_mask_H = np.zeros((num_elements,) + prediction_size_dict['H'])
    reg_labels_H = np.zeros((num_elements,) + prediction_size_dict['H'])
    reg_gen_M = np.zeros((num_elements,) + prediction_size_dict['M'])

    flow = np.zeros((num_elements, ndims) + prediction_size_dict['M'])

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(generator):

            H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
            H_md, H_m = data_dict['x_flo_mask'].to(device), data_dict['x_flo_init_mask'].to(device)
            M_md, M_m = data_dict['x_ref_mask'].to(device), data_dict['x_ref_mask'].to(device)
            M_l, H_l = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)

            if da_model is not None:
                nonlinear_field = data_dict['nonlinear'][0].to(device)
                affine = data_dict['affine'][0].to(device)

                H = da_model.transform(H, affine, nonlinear_field)
                H_m = da_model.transform(H_m, affine, nonlinear_field)
                H_md = da_model.transform(H_md, affine, nonlinear_field)

            M_g = model_dict['G_M'](H)
            rM_g, H_f, H_svf = model_dict['R_M'](M_g, M)

            if isinstance(model_dict['R_M'], models.VxmRigidDense):
                rH = model_dict['R_M'].predict_affine(H, H_f[0], inverse=False)
                rH = model_dict['R_M'].predict(rH, H_f[1], svf=False)

                rH_m = model_dict['R_M'].predict_affine(H_m, H_f[0], inverse=False)
                rH_m = model_dict['R_M'].predict(rH_m, H_f[1], svf=False)

                rH_l = model_dict['R_M'].predict_affine(H_l, H_f[0], inverse=False)
                rH_l = model_dict['R_M'].predict(rH_l, H_f[1], svf=False)

            else:

                rH = model_dict['R_M'].predict(H, H_f, svf=False)
                rH_m = model_dict['R_M'].predict(H_md, H_f, svf=False)
                rH_l = model_dict['R_M'].predict(H_l, H_f, svf=False)

            start = batch_idx * batch_size
            end = start + batch_size
            if batch_idx == num_batches - 1:
                end = num_elements

            data_H[start:end] = H[:, 0].cpu().detach().numpy()
            data_M[start:end] = M[:, 0].cpu().detach().numpy()
            mask_H[start:end] = H_md[:, 0].cpu().detach().numpy()
            mask_M[start:end] = M_m[:, 0].cpu().detach().numpy()
            labels_M[start:end] = np.argmax(M_l.cpu().detach().numpy(), axis=1)
            gen_M[start:end] = M_g[:,0].cpu().detach().numpy()

            reg_mask_H[start:end] = rH_m[:, 0].cpu().detach().numpy()
            reg_gen_M[start:end] = rM_g[:,0].cpu().detach().numpy()
            reg_data_H[start:end] = np.squeeze(rH.cpu().detach().numpy())
            reg_labels_H[start:end] = np.argmax(rH_l.cpu().detach().numpy(), axis=1)

            if isinstance(model_dict['R_M'], models.VxmRigidDense):
                flow[start:end] = np.squeeze(H_f[1].cpu().detach().numpy())
            else:
                flow[start:end] = np.squeeze(H_f.cpu().detach().numpy())


    return [data_M, data_H, reg_data_H], [mask_M, mask_H, reg_mask_H], [gen_M, reg_gen_M], [labels_M, reg_labels_H], flow

def predict_batch(data_dict, model_dict, prediction_size_dict, device, batch_size=1):
    with torch.no_grad():

        H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
        H_md, H_m = data_dict['x_flo_mask'].to(device), data_dict['x_flo_init_mask'].to(device)
        M_md, M_m = data_dict['x_ref_mask'].to(device), data_dict['x_ref_mask'].to(device)
        M_l, H_l = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)

        H, M = H.to(device), M.to(device)
        H_l = H_l.to(device)

        M_g = model_dict['G_M'](H)
        rM_g, H_f, H_svf = model_dict['R_M'](M_g, M)

        if isinstance(model_dict['R_M'], models.VxmRigidDense):
            rH = model_dict['R_M'].predict_affine(H, H_f[0], inverse=False)
            rH = model_dict['R_M'].predict(rH, H_f[1], svf=False)

            rH_m = model_dict['R_M'].predict_affine(H_m, H_f[0], inverse=False)
            rH_m = model_dict['R_M'].predict(rH_m, H_f[1], svf=False)

            rH_l = model_dict['R_M'].predict_affine(H_l, H_f[0], inverse=False)
            rH_l = model_dict['R_M'].predict(rH_l, H_f[1], svf=False)
        else:
            rH = model_dict['R_M'].predict(H, H_f, svf=False)
            rH_m = model_dict['R_M'].predict(H_m, H_f, svf=False)
            rH_l = model_dict['R_M'].predict(H_l, H_f, svf=False)

        data_H = np.squeeze(H[:, 0].cpu().detach().numpy())
        data_M = np.squeeze(M[:, 0].cpu().detach().numpy())
        mask_H = np.squeeze(M_m[:, 0].cpu().detach().numpy())
        mask_M = np.squeeze(M_m[:, 0].cpu().detach().numpy())
        labels_M =np.argmax(M_l.cpu().detach().numpy(), axis=1)
        gen_M = np.squeeze(M_g[:,0].cpu().detach().numpy())

        reg_mask_H = np.squeeze(rH_m[:, 0].cpu().detach().numpy())
        reg_gen_M = np.squeeze(rM_g[:, 0].cpu().detach().numpy())
        reg_data_H = np.squeeze(rH.cpu().detach().numpy())
        reg_labels_H = np.argmax(rH_l.cpu().detach().numpy(), axis=1)

        if isinstance(model_dict['R_M'], models.VxmRigidDense):
            flow = np.squeeze(H_f[1].cpu().detach().numpy())
        else:
            flow = np.squeeze(H_f.cpu().detach().numpy())


    return [data_M, data_H, reg_data_H], [mask_M, mask_H, reg_mask_H], [gen_M, reg_gen_M], [labels_M, reg_labels_H], flow

def predict_labels(generator, model_dict, prediction_size_dict, device):
    num_elements = len(generator.dataset)
    num_batches = len(generator)
    batch_size = generator.batch_size
    ndims = len(prediction_size_dict['M'])
    data_M = np.zeros((num_elements,) + prediction_size_dict['M'])
    data_H = np.zeros((num_elements,) + prediction_size_dict['H'])
    mask_M = np.zeros((num_elements,) + prediction_size_dict['M'])
    mask_H = np.zeros((num_elements,) + prediction_size_dict['H'])

    seg_M = np.zeros((num_elements,prediction_size_dict['latent']) + prediction_size_dict['M'])
    seg_H = np.zeros((num_elements,prediction_size_dict['latent']) + prediction_size_dict['H'])
    rseg_H = np.zeros((num_elements,prediction_size_dict['latent']) + prediction_size_dict['H'])

    flow = np.zeros((num_elements, ndims) + prediction_size_dict['M'])

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(generator):

            H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
            H_md, H_m = data_dict['x_flo_mask'].to(device), data_dict['x_flo_init_mask'].to(device)
            M_md, M_m = data_dict['x_ref_mask'].to(device), data_dict['x_ref_mask'].to(device)
            M_s, H_s = data_dict['x_ref_labels'].to(device), data_dict['x_flo_mask'].to(device)

            H, M = H.to(device), M.to(device)

            data_gen = model_dict['G_M'](torch.cat((H, M), dim=0))
            M_g = data_gen[:batch_size]

            rM_g, H_f, H_svf = model_dict['R_M'](M_g, M)

            if isinstance(model_dict['R_M'], models.VxmRigidDense):
                # rH = model_dict['R_M'].predict_affine(H, H_f[0], inverse=False)
                # rH = model_dict['R_M'].predict(rH, H_f[1], svf=False)

                rH_s = model_dict['R_M'].predict_affine(H_s, H_f[0], inverse=False)
                rH_s = model_dict['R_M'].predict(rH_s, H_f[1], svf=False)


            else:
                # rH = model_dict['R_M'].predict(H, H_f, svf=False)
                rH_s = model_dict['R_M'].predict(H_s, H_f, svf=False)

            start = batch_idx * batch_size
            end = start + batch_size
            if batch_idx == num_batches - 1:
                end = num_elements

            data_H[start:end] = H[:, 0].cpu().detach().numpy()
            data_M[start:end] = M[:, 0].cpu().detach().numpy()
            mask_H[start:end] = H_m[:, 0].cpu().detach().numpy()
            mask_M[start:end] = M_m[:, 0].cpu().detach().numpy()
            rseg_H[start:end] = rH_s.cpu().detach().numpy()
            seg_H[start:end] = H_s.cpu().detach().numpy()
            seg_M[start:end] = H_m.cpu().detach().numpy()

            if isinstance(model_dict['R_M'], models.VxmRigidDense):
                flow[start:end] = np.squeeze(H_f[1].cpu().detach().numpy())
            else:
                flow[start:end] = np.squeeze(H_f.cpu().detach().numpy())

    return [data_M, data_H], [mask_M, mask_H], [seg_M, seg_H, rseg_H], flow

def predict_batch_RoT(data_dict, model_dict, prediction_size_dict, device, batch_size=1):
    with torch.no_grad():

        H, M = data_dict['x_flo'].to(device), data_dict['x_ref'].to(device)
        H_md, H_m = data_dict['x_flo_mask'].to(device), data_dict['x_flo_init_mask'].to(device)
        M_md, M_m = data_dict['x_ref_mask'].to(device), data_dict['x_ref_mask'].to(device)
        M_l, H_l = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)

        H, M = H.to(device), M.to(device)
        H_l = H_l.to(device)

        rH, H_f, H_svf = model_dict['R_M'](H, M)
        M_g = model_dict['G_M'](H)
        rM_g = model_dict['G_M'](rH)

        rH_m = model_dict['R_M'].predict(H_m, H_f, svf=False)
        rH_l = model_dict['R_M'].predict(H_l, H_f, svf=False)

        data_H = np.squeeze(H[:, 0].cpu().detach().numpy())
        data_M = np.squeeze(M[:, 0].cpu().detach().numpy())
        mask_H = np.squeeze(M_m[:, 0].cpu().detach().numpy())
        mask_M = np.squeeze(M_m[:, 0].cpu().detach().numpy())
        labels_M =np.argmax(M_l.cpu().detach().numpy(), axis=1)
        gen_M = np.squeeze(M_g[:,0].cpu().detach().numpy())

        reg_mask_H = np.squeeze(rH_m[:, 0].cpu().detach().numpy())
        reg_gen_M = np.squeeze(rM_g[:, 0].cpu().detach().numpy())
        reg_data_H = np.squeeze(rH.cpu().detach().numpy())
        reg_labels_H = np.argmax(rH_l.cpu().detach().numpy(), axis=1)

        if isinstance(model_dict['R_M'], models.VxmRigidDense):
            flow = np.squeeze(H_f[1].cpu().detach().numpy())
        else:
            flow = np.squeeze(H_f.cpu().detach().numpy())


    return [data_M, data_H, reg_data_H], [mask_M, mask_H, reg_mask_H], [gen_M, reg_gen_M], [labels_M, reg_labels_H], flow


def predict_registration(data_dict, model, device, da_model=None, labels_flag=False):

    with torch.no_grad():
        ref_image, flo_image = data_dict['x_ref'].to(device), data_dict['x_flo'].to(device)
        ref_mask, flo_mask = data_dict['x_ref_mask'].to(device), data_dict['x_flo_mask'].to(device)
        ref_labels, flo_labels = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_dict['nonlinear']]
        affine_field = [aff.to(device) for aff in data_dict['affine']]

        if da_model is not None:
            flo_image_fake = da_model.transform(flo_image, affine_field[0], nonlinear_field[0])
            flo_mask_fake = da_model.transform(flo_mask, affine_field[0], nonlinear_field[0])
            flo_labels_fake = da_model.transform(flo_labels, affine_field[0], nonlinear_field[0])
        else:
            flo_image_fake = flo_image
            flo_mask_fake = flo_mask
            flo_labels_fake = flo_labels

        r, f, v = model(flo_image_fake, ref_image)
        f_rev = model.get_flow_field(-v)
        if isinstance(model, models.VxmRigidDense):
            r_flo = model.predict_affine(ref_image, f[0], inverse=True)
            r_flo = model.predict(r_flo, f_rev, svf=False)

            r_mask = model.predict_affine(flo_mask_fake, f[0], inverse=False)
            r_mask = model.predict(r_mask, f[1], svf=False)

            r_flo_mask = model.predict_affine(ref_mask, f[0], inverse=True)
            r_flo_mask = model.predict(r_flo_mask, f_rev, svf=False)

            r_labels = model.predict_affine(flo_labels_fake, f[0], inverse=False)
            r_labels = model.predict(r_labels, f[1], svf=False)

            flow = f[1].cpu().detach().numpy()

        else:
            r_mask = model.predict(flo_mask_fake, f, svf=False)
            r_labels = model.predict(flo_labels_fake, f, svf=False)

            r_flo = model.predict(ref_image, f_rev, svf=False)
            r_flo_mask = model.predict(ref_mask, f_rev, svf=False)
            flow = f.cpu().detach().numpy()

        ref_image = ref_image[:,0].cpu().detach().numpy()
        flo_image = flo_image_fake[:,0].cpu().detach().numpy()
        reg_image_ref = r[:,0].cpu().detach().numpy()
        reg_image_flo = r_flo[:,0].cpu().detach().numpy()

        ref_mask = ref_mask[:,0].cpu().detach().numpy()
        flo_mask = flo_mask_fake[:,0].cpu().detach().numpy()
        reg_mask_ref = r_mask[:,0].cpu().detach().numpy()
        reg_mask_flo = r_flo_mask[:,0].cpu().detach().numpy()
        reg_labels_ref = np.argmax(r_labels.cpu().detach().numpy(), axis=1)

    return ref_image, flo_image, reg_image_ref, reg_image_flo, \
           ref_mask, flo_mask, reg_mask_ref, reg_mask_flo, reg_labels_ref, flow

def predict_segmentation(data_dict, model, device, da_model=None):

    with torch.no_grad():
        ref_image = data_dict['x_ref'].to(device)
        ref_labels = data_dict['x_ref_labels'].to(device)
        ref_mask = data_dict['x_ref_mask'].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_dict['nonlinear']]
        affine_field = [aff.to(device) for aff in data_dict['affine']]

        if da_model is not None:
            ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0])
            ref_labels = da_model.transform(ref_labels, affine_field[0], nonlinear_field[0])

        ref_seg = model(ref_image)

        ref_image = ref_image[:,0].cpu().detach().numpy()
        ref_mask = ref_mask[:,0].cpu().detach().numpy()
        ref_labels = np.argmax(ref_labels.cpu().detach().numpy(), axis=1)
        ref_seg_prob = ref_seg.cpu().detach().numpy()
        ref_seg = np.argmax(ref_seg.cpu().detach().numpy(), axis=1)

    return ref_image, ref_mask, ref_labels, ref_seg, ref_seg_prob

