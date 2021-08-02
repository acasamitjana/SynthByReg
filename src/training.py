from os.path import join
import time
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src import models
from src.utils.visualization import plot_results
from src.callbacks import History, ModelCheckpoint, PrinterCallback, ToCSVCallback
from src.utils.visualization import slices

DOWNSAMPLING_DICT = {
    0: 0, #Padding
    4: 1, #Input size
    8: 2, #Downsamplig x2
    12: 3, #Downsamplig x4
    16: 4, #Downsamplig x8
    20: 5, #Downsamplig x8
    24: 6, #Downsamplig x8

}

class InfoNCE(object):

    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model, nce_layers, num_patches,
                 clip_grad, dx_lr, p_dict, d_iter=1, cycle='reg', mask_nce_flag=True):
        self.device = device
        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss_G', 'loss_D', 'loss', 'time_duration (s)']

        attach = True if p_dict['STARTING_EPOCH'] > 0 else False
        logger = History(self.log_keys)
        mcheck = ModelCheckpoint(join(p_dict['RESULTS_DIR'], 'checkpoints'), p_dict['SAVE_MODEL_FREQUENCY'])
        training_printer = PrinterCallback()
        training_tocsv = ToCSVCallback(filepath=join(p_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                                       keys=self.log_keys, attach=attach)
        callback_list = [logger, mcheck, training_printer, training_tocsv]

        self.callbacks = callback_list + callbacks

        self.da_model = da_model if da_model is not None else False

        self.nce_layers = nce_layers
        self.num_patches = num_patches

        self.cycle = cycle
        self.mask_nce_flag = mask_nce_flag

        self.clip_grad = clip_grad
        self.dx_lr = dx_lr
        self.d_iter = d_iter
        self.parameter_dict = p_dict


    def compute_NCE_loss(self, data, model_dict, masked=True):
        raise NotImplementedError

    def initialize_F(self, tensor_dict, model_dict, optimizer_dict, log_dict={}, weightsfile=None):

        '''
        Networks A/B are defined on the domain they apply (e.g: G_A transforms from B to A, while DX_A is used to
        discriminate between real/fake in domain A (generated A and real A))
        '''


        D_loss = self.compute_D_loss(tensor_dict, model_dict, {})
        D_loss.backward()
        G_loss = self.compute_G_loss(tensor_dict, model_dict, {})
        G_loss.backward()

        if weightsfile is not None:
            checkpoint = torch.load(weightsfile)
            for model_key, model in model_dict.items():
                if model_key == 'F_M':
                    model.load_state_dict(checkpoint['state_dict_' + model_key])

        elif self.parameter_dict['STARTING_EPOCH'] > 0:
            weightsfile = 'model_checkpoint.' + str(self.parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
            checkpoint = torch.load(join(self.parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
            for optimizer_key, optimizer in optimizer_dict.items():
                if optimizer_key == 'F_M':
                    optimizer.load_state_dict(checkpoint['optimizer_' + optimizer_key])

            for model_key, model in model_dict.items():
                if model_key == 'F_M':
                    model.load_state_dict(checkpoint['state_dict_' + model_key])



        if self.loss_weight_dict['nce']:
            optimizer_F_M = torch.optim.Adam(model_dict['F_M'].parameters(), lr=self.dx_lr, betas=(0.5, 0.999))
            optimizer_dict['F_M'] = optimizer_F_M

    def compute_D_loss(self, tensor_dict, model_dict, log_dict={}):

        data_M, generated = tensor_dict['data_M'], tensor_dict['gen_M']
        gen_D = generated.detach()

        pred_labels_M = model_dict['D_M'](data_M)
        pred_labels_gen_M = model_dict['D_M'](gen_D)

        # Adversarial loss
        DX_loss_M_real = self.loss_function_dict['gan'](pred_labels_M, True).mean()
        DX_loss_M_fake = self.loss_function_dict['gan'](pred_labels_gen_M, False).mean()
        D_loss = (DX_loss_M_fake + DX_loss_M_real) * 0.5

        log_dict['loss_D'] = D_loss.item()

        return D_loss

    def compute_G_loss(self,tensor_dict, model_dict, log_dict={}):

        pred_labels_gen_M = model_dict['D_M'](tensor_dict['gen_M'])
        GAN_loss_M = self.loss_function_dict['gan'](pred_labels_gen_M, True).mean()
        GAN_loss = self.loss_weight_dict['gan'] * GAN_loss_M
        log_dict['loss_' + self.loss_function_dict['gan'].name] = GAN_loss.item()

        ##### Registration
        REG_loss_NCC = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_ncc'].name] = REG_loss_NCC
        if self.loss_weight_dict['registration_ncc'] > 0:
            REG_loss_NCC_M = self.loss_function_dict['registration_ncc'](tensor_dict['data_M'], tensor_dict['gen_reg_M'], mask=tensor_dict['mask_M_d'])
            REG_loss_NCC = self.loss_weight_dict['registration_ncc'] * REG_loss_NCC_M
            log_dict['loss_' + self.loss_function_dict['registration_ncc'].name] = REG_loss_NCC.item()

        REG_loss_l1 = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = REG_loss_l1
        if self.loss_weight_dict['registration_l1'] > 0:
            REG_loss_M = self.loss_function_dict['registration_l1'](tensor_dict['data_M'], tensor_dict['gen_reg_M'], mask=tensor_dict['mask_M_d'])
            if 'reg_M' in tensor_dict.keys():
                REG_loss_M += self.loss_function_dict['registration_l1'](tensor_dict['gen_M'], tensor_dict['reg_M'], mask=tensor_dict['mask_H_d'])
                REG_loss_M = 0.5*REG_loss_M
            REG_loss_l1 = self.loss_weight_dict['registration_l1'] * REG_loss_M
            log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = REG_loss_l1.item()

        REGs_loss = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss
        if self.loss_weight_dict['registration_smoothness'] > 0:
            REGs_loss_M = self.loss_function_dict['registration_smoothness'](tensor_dict['velocity_field_M'])
            REGs_loss = self.loss_weight_dict['registration_smoothness'] * REGs_loss_M
            log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss.item()

        NCE_loss = 0.0
        log_dict['loss_nce'] = NCE_loss
        if self.loss_weight_dict['nce'] > 0:
            if self.cycle == 'idt':
                ####### Cambiar això per Allen, que no vull CycleGAN!!!!
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                ####### Cambiar això per Allen, que no vull CycleGAN!!!!
                # data_in = tensor_dict['reg_data_H'], tensor_dict['gen_reg_M'], tensor_dict['mask_M']
                # NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                data_in = tensor_dict['data_M'], tensor_dict['idt_M'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X + NCE_loss_Y

            elif self.cycle == 'reg':
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                data_in = tensor_dict['data_M'], tensor_dict['reg_data_H'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X + NCE_loss_Y

            elif self.cycle == 'reg_2':
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                data_in = tensor_dict['reg_data_H'], tensor_dict['gen_reg_M'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X + NCE_loss_Y


            elif self.cycle == 'reg_only':
                data_in = tensor_dict['data_M'], tensor_dict['reg_data_H'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_Y

            elif self.cycle == 'reg_only_2':
                data_in = tensor_dict['reg_data_H'], tensor_dict['gen_reg_M'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_Y

            else:
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X

            log_dict['loss_nce'] = NCE_loss.item()

        # Generator
        GEN_loss = GAN_loss + NCE_loss + REG_loss_NCC + REG_loss_l1 + REGs_loss
        log_dict['loss_G'] = GEN_loss.item()
        return GEN_loss

    def iterate_weakly(self, model_dict, optimizer_dict, generator_train, epoch, **kwargs):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}
            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H = data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            labels_M, labels_H = data_dict['x_ref_labels'].to(self.device), data_dict['x_flo_labels'].to(self.device)

            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0  # np.random.rand(1)
                flipud = 0  # np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr, mode='nearest')
                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr,
                                                   mode='nearest')
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr, mode='nearest')
                labels_M = self.da_model.transform(labels_M, affine[1], nonlinear_field[1], flipud, fliplr,
                                                   mode='nearest')
                labels_H = self.da_model.transform(labels_H, affine[0], nonlinear_field[0], flipud, fliplr,
                                                   mode='nearest')

            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d,
                'labels_M': labels_M
            }

            ###################
            #  Initialization #
            ###################

            if batch_idx == 0 and epoch == 0:
                if self.cycle == 'idt':
                    generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                    gen_M = generated_data[:batch_size]
                    idt_M = generated_data[batch_size:]
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    tensor_dict['idt_M'] = idt_M

                else:
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                reg_labels_H = model_dict['R_M'].predict(labels_H, flow_field_M, svf=False)
                tensor_dict['gen_M'] = gen_M
                tensor_dict['gen_reg_M'] = gen_reg_M
                tensor_dict['velocity_field_M'] = velocity_field_M
                tensor_dict['reg_labels_H'] = reg_labels_H

                self.initialize_F(tensor_dict, model_dict, optimizer_dict)

            ##################
            #  Forward pass #
            ##################

            if self.cycle == 'idt':
                generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                gen_M = generated_data[:batch_size]
                idt_M = generated_data[batch_size:]
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                tensor_dict['idt_M'] = idt_M

            else:
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            reg_labels_H = model_dict['R_M'].predict(labels_H, flow_field_M, svf=False)
            tensor_dict['gen_M'] = gen_M
            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M
            tensor_dict['reg_labels_H'] = reg_labels_H

            ##################
            #  Discriminator #
            ##################

            for p in model_dict['D_M'].parameters():
                p.requires_grad = True  # to avoid computation

            for _ in range(self.d_iter):
                optimizer_dict['D_M'].zero_grad()
                D_loss = self.compute_D_loss(tensor_dict, model_dict, log_dict)
                D_loss.backward()
                if self.clip_grad: torch.nn.utils.clip_grad_norm_(model_dict['D_M'].parameters(), self.clip_grad)
                optimizer_dict['D_M'].step()

            ##############
            #  Generator #
            ##############
            for p in model_dict['D_M'].parameters():
                p.requires_grad = False  # to avoid computation

            optimizer_dict['G_M'].zero_grad()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            optimizer_dict['G_M'].step()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].step()

            log_dict['loss'] = log_dict['loss_G'] + log_dict['loss_D']

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def iterate(self, model_dict, optimizer_dict, generator_train, epoch, **kwargs):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H = data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0#np.random.rand(1)
                flipud = 0#np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr)

            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d
            }

            ###################
            #  Initialization #
            ###################

            if batch_idx == 0 and epoch == 0:
                if self.cycle == 'idt':
                    generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                    gen_M = generated_data[:batch_size]
                    idt_M = generated_data[batch_size:]
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    tensor_dict['idt_M'] = idt_M

                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                else:
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

                tensor_dict['gen_M'] = gen_M
                tensor_dict['gen_reg_M'] = gen_reg_M
                tensor_dict['velocity_field_M'] = velocity_field_M

                self.initialize_F(tensor_dict, model_dict, optimizer_dict)

            ##################
            #  Forward pass #
            ##################

            if self.cycle == 'idt':
                generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                gen_M = generated_data[:batch_size]
                idt_M = generated_data[batch_size:]
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                tensor_dict['idt_M'] = idt_M

                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            else:
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

            tensor_dict['gen_M'] = gen_M
            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M

            ##################
            #  Discriminator #
            ##################

            for p in model_dict['D_M'].parameters():
                p.requires_grad = True  # to avoid computation

            for _ in range(self.d_iter):
                optimizer_dict['D_M'].zero_grad()
                D_loss = self.compute_D_loss(tensor_dict, model_dict, log_dict)
                D_loss.backward()
                if self.clip_grad: torch.nn.utils.clip_grad_norm_(model_dict['D_M'].parameters(), self.clip_grad)
                optimizer_dict['D_M'].step()

            ##############
            #  Generator #
            ##############
            for p in model_dict['D_M'].parameters():
                p.requires_grad = False  # to avoid computation

            optimizer_dict['G_M'].zero_grad()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            optimizer_dict['G_M'].step()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].step()

            log_dict['loss'] = log_dict['loss_G'] + log_dict['loss_D']

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def iterate_bidir(self, model_dict, optimizer_dict, generator_train, epoch, **kwargs):

        '''
        '''
        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H_d, mask_H = data_dict['x_flo_mask'].to(self.device),data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0#np.random.rand(1)
                flipud = 0#np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H_d = self.da_model.transform(mask_H_d, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr)

            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d,
                'mask_H_d': mask_H_d
            }

            ###################
            #  Initialization #
            ###################

            if batch_idx == 0 and epoch == 0:
                if self.cycle == 'idt':
                    generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                    gen_M = generated_data[:batch_size]
                    idt_M = generated_data[batch_size:]
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    tensor_dict['idt_M'] = idt_M

                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                else:
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

                flow_image_rev = model_dict['R_M'].get_flow_field(-velocity_field_M)
                reg_M = model_dict['R_M'].predict(data_M, flow_image_rev, svf=False)

                tensor_dict['gen_M'] = gen_M
                tensor_dict['gen_reg_M'] = gen_reg_M
                tensor_dict['velocity_field_M'] = velocity_field_M
                tensor_dict['reg_M'] = reg_M

                self.initialize_F(tensor_dict, model_dict, optimizer_dict)

            ##################
            #  Forward pass #
            ##################

            if self.cycle == 'idt':
                generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                gen_M = generated_data[:batch_size]
                idt_M = generated_data[batch_size:]
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                tensor_dict['idt_M'] = idt_M

                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            else:
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

            flow_image_rev = model_dict['R_M'].get_flow_field(-velocity_field_M)
            reg_M = model_dict['R_M'].predict(data_M, flow_image_rev, svf=False)

            tensor_dict['gen_M'] = gen_M
            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M
            tensor_dict['reg_M'] = reg_M
            ##################
            #  Discriminator #
            ##################

            for p in model_dict['D_M'].parameters():
                p.requires_grad = True  # to avoid computation

            for _ in range(self.d_iter):
                optimizer_dict['D_M'].zero_grad()
                D_loss = self.compute_D_loss(tensor_dict, model_dict, log_dict)
                D_loss.backward()
                if self.clip_grad: torch.nn.utils.clip_grad_norm_(model_dict['D_M'].parameters(), self.clip_grad)
                optimizer_dict['D_M'].step()

            ##############
            #  Generator #
            ##############
            for p in model_dict['D_M'].parameters():
                p.requires_grad = False  # to avoid computation

            optimizer_dict['G_M'].zero_grad()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            optimizer_dict['G_M'].step()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].step()

            log_dict['loss'] = log_dict['loss_G'] + log_dict['loss_D']

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def train(self, model_dict, optimizer_dict, generator_train, weakly_flag, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        logs_dict = {}
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            model_dict['G_M'].train()
            model_dict['D_M'].train()
            model_dict['F_M'].train()

            if weakly_flag:
                self.iterate_weakly(model_dict, optimizer_dict, generator_train, epoch, **kwargs)
            else:
                self.iterate(model_dict, optimizer_dict, generator_train, epoch, **kwargs)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)

        print('All done')

class InfoNCENoGAN(InfoNCE):

    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model, nce_layers, num_patches,
                 clip_grad, dx_lr, p_dict, d_iter=1, cycle='reg', mask_nce_flag=True):
        super().__init__(device, loss_function_dict, loss_weight_dict, callbacks, da_model, nce_layers, num_patches,
                         clip_grad, dx_lr, p_dict, d_iter, cycle, mask_nce_flag)
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss', 'time_duration (s)']

    def initialize_F(self, tensor_dict, model_dict, optimizer_dict, log_dict={}, weightsfile=None):

        '''
        Networks A/B are defined on the domain they apply (e.g: G_A transforms from B to A, while DX_A is used to
        discriminate between real/fake in domain A (generated A and real A))
        '''


        G_loss = self.compute_G_loss(tensor_dict, model_dict, {})
        G_loss.backward()

        if weightsfile is not None:
            print(weightsfile)
            checkpoint = torch.load(weightsfile)
            for model_key, model in model_dict.items():
                if model_key == 'F_M':
                    model.load_state_dict(checkpoint['state_dict_' + model_key])

        elif self.parameter_dict['STARTING_EPOCH'] > 0:
            weightsfile = 'model_checkpoint.' + str(self.parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
            checkpoint = torch.load(join(self.parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
            for optimizer_key, optimizer in optimizer_dict.items():
                if optimizer_key == 'F_M':
                    optimizer.load_state_dict(checkpoint['optimizer_' + optimizer_key])

            for model_key, model in model_dict.items():
                if model_key == 'F_M':
                    model.load_state_dict(checkpoint['state_dict_' + model_key])

        if self.loss_weight_dict['nce']:
            optimizer_F_M = torch.optim.Adam(model_dict['F_M'].parameters(), lr=self.dx_lr, betas=(0.5, 0.999))
            optimizer_dict['F_M'] = optimizer_F_M

    def compute_G_loss(self, tensor_dict, model_dict, log_dict={}):

        ##### Registration
        REG_loss_NCC = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_ncc'].name] = REG_loss_NCC
        if self.loss_weight_dict['registration_ncc'] > 0:
            REG_loss_NCC_M = self.loss_function_dict['registration_ncc'](tensor_dict['data_M'], tensor_dict['gen_reg_M'], mask=tensor_dict['mask_M_d'])
            REG_loss_NCC = self.loss_weight_dict['registration_ncc'] * REG_loss_NCC_M
            log_dict['loss_' + self.loss_function_dict['registration_ncc'].name] = REG_loss_NCC.item()

        REG_loss_l1 = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = REG_loss_l1
        if self.loss_weight_dict['registration_l1'] > 0:
            REG_loss_M = self.loss_function_dict['registration_l1'](tensor_dict['data_M'], tensor_dict['gen_reg_M'], mask=tensor_dict['mask_M_d'])
            REG_loss_l1 = self.loss_weight_dict['registration_l1'] * REG_loss_M
            log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = REG_loss_l1.item()

        REGs_loss = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss
        if self.loss_weight_dict['registration_smoothness'] > 0:
            REGs_loss_M = self.loss_function_dict['registration_smoothness'](tensor_dict['velocity_field_M'])
            REGs_loss = self.loss_weight_dict['registration_smoothness'] * REGs_loss_M
            log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss.item()

        NCE_loss = 0.0
        log_dict['loss_nce'] = NCE_loss
        if self.loss_weight_dict['nce'] > 0:
            if self.cycle == 'idt':
                ####### Cambiar això per Allen, que no vull CycleGAN!!!!
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                ####### Cambiar això per Allen, que no vull CycleGAN!!!!
                # data_in = tensor_dict['reg_data_H'], tensor_dict['gen_reg_M'], tensor_dict['mask_M']
                # NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                data_in = tensor_dict['data_M'], tensor_dict['idt_M'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X + NCE_loss_Y

            elif self.cycle == 'reg':
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                data_in = tensor_dict['data_M'], tensor_dict['reg_data_H'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X + NCE_loss_Y

            elif self.cycle == 'reg_2':
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)

                data_in = tensor_dict['reg_data_H'], tensor_dict['gen_reg_M'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X + NCE_loss_Y


            elif self.cycle == 'reg_only':
                data_in = tensor_dict['data_M'], tensor_dict['reg_data_H'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_Y

            elif self.cycle == 'reg_only_2':
                data_in = tensor_dict['reg_data_H'], tensor_dict['gen_reg_M'], tensor_dict['mask_M']
                NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_Y

            else:
                data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
                NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=self.mask_nce_flag)
                NCE_loss = NCE_loss_X

            log_dict['loss_nce'] = NCE_loss.item()

        # Generator
        GEN_loss = NCE_loss + REG_loss_NCC + REG_loss_l1 + REGs_loss
        log_dict['loss'] = GEN_loss.item()
        return GEN_loss

    def iterate(self, model_dict, optimizer_dict, generator_train, epoch, **kwargs):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H_d, mask_H = data_dict['x_flo_mask'].to(self.device), data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0#np.random.rand(1)
                flipud = 0#np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr)


            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d,
            }

            ###################
            #  Initialization #
            ###################

            if batch_idx == 0 and epoch == 0:
                if self.cycle == 'idt':
                    generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                    gen_M = generated_data[:batch_size]
                    idt_M = generated_data[batch_size:]
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    tensor_dict['idt_M'] = idt_M

                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                else:
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

                tensor_dict['gen_M'] = gen_M
                tensor_dict['gen_reg_M'] = gen_reg_M
                tensor_dict['velocity_field_M'] = velocity_field_M

                self.initialize_F(tensor_dict, model_dict, optimizer_dict)

            ##################
            #  Forward pass #
            ##################

            if self.cycle == 'idt':
                generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                gen_M = generated_data[:batch_size]
                idt_M = generated_data[batch_size:]
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                tensor_dict['idt_M'] = idt_M

                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            else:
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

            tensor_dict['gen_M'] = gen_M
            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M

            ##############
            #  Generator #
            ##############

            optimizer_dict['G_M'].zero_grad()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            optimizer_dict['G_M'].step()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].step()

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def train(self, model_dict, optimizer_dict, generator_train, weakly_flag, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        logs_dict = {}
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            model_dict['G_M'].train()
            model_dict['F_M'].train()

            if weakly_flag:
                self.iterate_weakly(model_dict, optimizer_dict, generator_train, epoch, **kwargs)
            else:
                self.iterate(model_dict, optimizer_dict, generator_train, epoch, **kwargs)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)

        print('All done')

class InfoNCE2D(InfoNCE):

    def compute_NCE_loss(self, data, model_dict, masked=True):
        src, trgt = data[:2]
        n_layers = len(self.nce_layers)

        feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
        feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
        if masked:
            src_m = data[2]
            mp = nn.ReflectionPad2d(3)
            mm = nn.MaxPool2d(2)
            kernel = torch.ones((1, 1, 3, 3), device=self.device, requires_grad=False)

            downsampling_masks = [mp(src_m), src_m]

            for it in range(2, 5):
                m_tmp = (F.conv2d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                downsampling_masks.append(mm(m_tmp))
            for it in range(5, 6):
                m_tmp = (F.conv2d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                downsampling_masks.append(m_tmp)

            feat_src_m = []
            for layer in self.nce_layers:
                feat_src_m.append(downsampling_masks[DOWNSAMPLING_DICT[layer]])

            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None,
                                                         mask_sampling=feat_src_m)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        else:
            feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
            feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        NCE_loss = 0.0
        for f_q, f_k, crit in zip(feat_trgt_pool, feat_src_pool, self.loss_function_dict['nce']):
            loss = crit(f_q, f_k) * self.loss_weight_dict['nce']
            NCE_loss += loss.mean()
        return NCE_loss / n_layers

class InfoNCE2DNoGAN(InfoNCE2D, InfoNCENoGAN):
    pass

class InfoNCE2DNoGANJoint(InfoNCE2DNoGAN):

    def iterate(self, model_dict, optimizer_dict, generator_train, epoch, train_R, weightsfile, **kwargs):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H_d, mask_H = data_dict['x_flo_mask'].to(self.device), data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0#np.random.rand(1)
                flipud = 0#np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr)


            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d,
            }

            ###################
            #  Initialization #
            ###################

            if batch_idx == 0 and epoch == 0:
                if self.cycle == 'idt':
                    generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                    gen_M = generated_data[:batch_size]
                    idt_M = generated_data[batch_size:]
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    tensor_dict['idt_M'] = idt_M

                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                    reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                    tensor_dict['reg_data_H'] = reg_data_H

                else:
                    gen_M = model_dict['G_M'](data_H)
                    gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

                tensor_dict['gen_M'] = gen_M
                tensor_dict['gen_reg_M'] = gen_reg_M
                tensor_dict['velocity_field_M'] = velocity_field_M

                self.initialize_F(tensor_dict, model_dict, optimizer_dict, weightsfile=weightsfile)

            ##################
            #  Forward pass #
            ##################

            if self.cycle == 'idt':
                generated_data = model_dict['G_M'](torch.cat((data_H, data_M), dim=0))
                gen_M = generated_data[:batch_size]
                idt_M = generated_data[batch_size:]
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                tensor_dict['idt_M'] = idt_M

                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            elif self.cycle == 'reg' or self.cycle == 'reg_only' or self.cycle == 'reg_only_2' or self.cycle == 'reg_2':
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)
                reg_data_H = model_dict['R_M'].predict(data_H, flow_field_M, svf=False)
                tensor_dict['reg_data_H'] = reg_data_H

            else:
                gen_M = model_dict['G_M'](data_H)
                gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

            tensor_dict['gen_M'] = gen_M
            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M

            ##############
            #  Generator #
            ##############

            if train_R: optimizer_dict['R_M'].zero_grad()
            optimizer_dict['G_M'].zero_grad()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            if train_R: optimizer_dict['R_M'].step()
            optimizer_dict['G_M'].step()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].step()

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def train(self, model_dict, optimizer_dict, generator_train, weakly_flag, bidir_flag, epoch_unfreeze=0, weightsfile=None, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        logs_dict = {}
        train_R = False
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            model_dict['G_M'].train()
            model_dict['F_M'].train()
            if epoch >= epoch_unfreeze:
                for param in model_dict['R_M'].parameters():
                    param.requires_grad = True
                model_dict['R_M'].train()
                train_R = True

            self.iterate(model_dict, optimizer_dict, generator_train, epoch, train_R, weightsfile,**kwargs)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)

        print('All done')

    def iterate_R(self, model_dict, optimizer_dict, generator_train, epoch):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H_d, mask_H = data_dict['x_flo_mask'].to(self.device), data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0#np.random.rand(1)
                flipud = 0#np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)


            tensor_dict = {
                'data_M': data_M,
                'mask_M_d': mask_M_d,
            }

            ##################
            #  Forward pass #
            ##################

            gen_M = model_dict['G_M'](data_H)
            gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M

            ##############
            #  Generator #
            ##############

            optimizer_dict['R_M'].zero_grad()
            R_loss = self.compute_R_loss(tensor_dict, log_dict)
            R_loss.backward()

            optimizer_dict['R_M'].step()

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def compute_R_loss(self, tensor_dict, log_dict={}):


        R_loss = self.loss_function_dict['registration_l1'](tensor_dict['data_M'], tensor_dict['gen_reg_M'], mask=tensor_dict['mask_M_d'])
        Rs_loss = self.loss_function_dict['registration_smoothness'](tensor_dict['velocity_field_M'])
        log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = R_loss.item()
        log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = Rs_loss.item()

        # Generator
        Reg_loss = R_loss + Rs_loss
        log_dict['loss'] = Reg_loss.item()
        return Reg_loss

    def train_R(self, model_dict, optimizer_dict, generator_train, weakly_flag, bidir_flag, epoch_unfreeze=0, weightsfile=None, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        logs_dict = {}
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            for param in model_dict['R_M'].parameters():
                param.requires_grad = True
            model_dict['R_M'].train()

            self.iterate_R(model_dict, optimizer_dict, generator_train, epoch)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)

        print('All done')

class InfoNCE3D(InfoNCE):

    # def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model, nce_layers, num_patches,
    #              clip_grad, dx_lr, p_dict, d_iter=1, cycle='reg'):
    #     super().__init__(device, loss_function_dict, loss_weight_dict, callbacks, da_model, nce_layers, num_patches,
    #                      clip_grad, dx_lr, p_dict, d_iter, cycle)
    #     self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss', 'time_duration (s)']

    def compute_NCE_loss(self,data, model_dict, masked=True):
        src, trgt = data[:2]
        n_layers = len(self.nce_layers)

        feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
        feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
        if masked:
            with torch.no_grad():
                src_m = data[2]
                mp = nn.ConstantPad3d(3, -1.)
                mm = nn.MaxPool3d(2)
                kernel = torch.ones((1, 1, 3, 3, 3), device=self.device, requires_grad=False)

                downsampling_masks = [mp(src_m), src_m]

                for it in range(2, 5):
                    m_tmp = (F.conv3d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                    downsampling_masks.append(mm(m_tmp))
                for it in range(5, 6):
                    m_tmp = (F.conv3d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                    downsampling_masks.append(m_tmp)

                feat_src_m = []
                for layer in self.nce_layers:
                    feat_src_m.append(downsampling_masks[DOWNSAMPLING_DICT[layer]])

            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None,
                                                         mask_sampling=feat_src_m)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        else:
            feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
            feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        NCE_loss = 0.0
        for f_q, f_k, crit in zip(feat_trgt_pool, feat_src_pool, self.loss_function_dict['nce']):
            loss = crit(f_q, f_k) * self.loss_weight_dict['nce']
            NCE_loss += loss.mean()
        return NCE_loss / n_layers

    # def initialize_F(self, tensor_dict, model_dict, optimizer_dict, log_dict={}):
    #
    #     '''
    #     Networks A/B are defined on the domain they apply (e.g: G_A transforms from B to A, while DX_A is used to
    #     discriminate between real/fake in domain A (generated A and real A))
    #     '''
    #
    #     D_loss = self.compute_D_loss(tensor_dict, model_dict, {})
    #     D_loss.backward()
    #     G_loss = self.compute_G_loss(tensor_dict, model_dict, {})
    #     G_loss.backward()
    #
    #     if self.parameter_dict['STARTING_EPOCH'] > 0:
    #         weightsfile = 'model_checkpoint.' + str(self.parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
    #         checkpoint = torch.load(join(self.parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
    #         for optimizer_key, optimizer in optimizer_dict.items():
    #             if optimizer_key == 'F_M':
    #                 optimizer.load_state_dict(checkpoint['optimizer_' + optimizer_key])
    #
    #         for model_key, model in model_dict.items():
    #             if model_key == 'F_M':
    #                 model.load_state_dict(checkpoint['state_dict_' + model_key])
    #
    #     if self.loss_weight_dict['nce']:
    #         optimizer_F_M = torch.optim.Adam(model_dict['F_M'].parameters(), lr=self.dx_lr, betas=(0.5, 0.999))
    #         optimizer_dict['F_M'] = optimizer_F_M
    #
    # def compute_D_loss(self, tensor_dict, model_dict, log_dict={}):
    #
    #     data_M, gen_M = tensor_dict['data_M'], tensor_dict['gen_M']
    #     gen_D = gen_M.detach()
    #
    #     pred_labels_M = model_dict['D_M'](data_M)
    #     pred_labels_gen_M = model_dict['D_M'](gen_D)
    #
    #     # Adversarial loss
    #     DX_loss_M_real = self.loss_function_dict['gan'](pred_labels_M, True).mean()
    #     DX_loss_M_fake = self.loss_function_dict['gan'](pred_labels_gen_M, False).mean()
    #     D_loss = (DX_loss_M_fake + DX_loss_M_real) * 0.5
    #
    #     log_dict['loss_D'] = D_loss.item()
    #
    #     return D_loss
    #
    # def compute_G_loss(self, tensor_dict, model_dict, log_dict={}):
    #
    #     # GAN
    #     pred_labels_gen_M = model_dict['D_M'](tensor_dict['gen_M'])
    #     GAN_loss_M = self.loss_function_dict['gan'](pred_labels_gen_M, True).mean()
    #     GAN_loss = self.loss_weight_dict['gan'] * GAN_loss_M
    #     log_dict['loss_' + self.loss_function_dict['gan'].name] = GAN_loss.item()
    #
    #     # Registration
    #     REG_loss_LAB = 0.0
    #     log_dict['loss_' + self.loss_function_dict['registration_labels'].name] = REG_loss_LAB
    #     if self.loss_weight_dict['registration_labels'] > 0:
    #         labels_M, reg_labels_H = tensor_dict['labels_M'], tensor_dict['reg_labels_H']
    #         REG_loss_LAB = self.loss_function_dict['registration_labels'](labels_M, reg_labels_H)
    #         REG_loss_LAB = self.loss_weight_dict['registration_labels'] * REG_loss_LAB
    #         log_dict['loss_' + self.loss_function_dict['registration_labels'].name] = REG_loss_LAB.item()
    #
    #     REG_loss_l1 = 0.0
    #     log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = REG_loss_l1
    #     if self.loss_weight_dict['registration_l1'] > 0:
    #         data_M, gen_reg_M, mask_M_d = tensor_dict['data_M'], tensor_dict['gen_reg_M'], tensor_dict['mask_M_d']
    #         REG_loss_M = self.loss_function_dict['registration_l1'](data_M, gen_reg_M, mask=mask_M_d)
    #         REG_loss_l1 = self.loss_weight_dict['registration_l1'] * REG_loss_M
    #         log_dict['loss_' + self.loss_function_dict['registration_l1'].name] = REG_loss_l1.item()
    #
    #     REGs_loss = 0.0
    #     log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss
    #     if self.loss_weight_dict['registration_smoothness'] > 0:
    #         velocity_field_M = tensor_dict['velocity_field_M']
    #         REGs_loss_M = self.loss_function_dict['registration_smoothness'](velocity_field_M)
    #         REGs_loss = self.loss_weight_dict['registration_smoothness'] * REGs_loss_M
    #         log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss.item()
    #
    #     NCE_loss = 0.0
    #     log_dict['loss_nce'] = NCE_loss
    #     if self.loss_weight_dict['nce'] > 0:
    #         data_in = tensor_dict['data_H'], tensor_dict['gen_M'], tensor_dict['mask_H']
    #         NCE_loss_X = self.compute_NCE_loss(data_in, model_dict, masked=True)
    #
    #         if self.cycle == 'reg':
    #             data_in = tensor_dict['data_M'], tensor_dict['reg_data_H'], tensor_dict['mask_M']
    #             NCE_loss_Y = self.compute_NCE_loss(data_in, model_dict, masked=True)
    #             NCE_loss = NCE_loss_X + NCE_loss_Y
    #         else:
    #             NCE_loss = NCE_loss_X
    #         log_dict['loss_nce'] = NCE_loss.item()
    #
    #     # Generator
    #     GEN_loss = GAN_loss +  NCE_loss + REG_loss_LAB + REG_loss_l1 + REGs_loss
    #     log_dict['loss_G'] = GEN_loss.item()
    #     return GEN_loss
    #
    # def train(self, model_dict, optimizer_dict, generator_train, weakly_flag, **kwargs):
    #     for cb in self.callbacks:
    #         cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])
    #
    #     logs_dict = {}
    #     for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):
    #
    #         epoch_start_time = time.time()
    #         for cb in self.callbacks:
    #             cb.on_epoch_init(model_dict, epoch)
    #
    #         model_dict['G_M'].train()
    #         model_dict['D_M'].train()
    #         model_dict['F_M'].train()
    #
    #         if weakly_flag:
    #             self.iterate_weakly(model_dict, optimizer_dict, generator_train, epoch, **kwargs)
    #         else:
    #             self.iterate(model_dict, optimizer_dict, generator_train, epoch, **kwargs)
    #
    #         epoch_end_time = time.time()
    #         logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time
    #
    #         for cb in self.callbacks:
    #             cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)
    #
    #     for cb in self.callbacks:
    #         cb.on_train_fi(model_dict)
    #
    #     plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)
    #
    #     print('All done')

class InfoNCENoGAN3D(InfoNCENoGAN, InfoNCE3D):

    pass


class RoT(object):

    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model,
                 clip_grad, dx_lr, p_dict, d_iter=1):
        self.device = device
        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss_G', 'loss_D', 'loss', 'time_duration (s)']

        attach = True if p_dict['STARTING_EPOCH'] > 0 else False
        logger = History(self.log_keys)
        mcheck = ModelCheckpoint(join(p_dict['RESULTS_DIR'], 'checkpoints'), p_dict['SAVE_MODEL_FREQUENCY'])
        training_printer = PrinterCallback()
        training_tocsv = ToCSVCallback(filepath=join(p_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                                       keys=self.log_keys, attach=attach)
        callback_list = [logger, mcheck, training_printer, training_tocsv]

        self.callbacks = callback_list + callbacks

        self.da_model = da_model if da_model is not None else False

        self.clip_grad = clip_grad
        self.dx_lr = dx_lr
        self.d_iter = d_iter
        self.parameter_dict = p_dict

    def compute_D_loss(self, tensor_dict, model_dict, log_dict={}):

        data_M, generated_RT, generated_TR = tensor_dict['data_M'], tensor_dict['gen_M_RT'], tensor_dict['gen_M_TR']
        gen_D_TR = generated_TR.detach()
        gen_D_RT = generated_RT.detach()

        pred_labels_M = model_dict['D_M'](data_M)
        pred_labels_gen_M_TR = model_dict['D_M'](gen_D_TR)
        pred_labels_gen_M_RT = model_dict['D_M'](gen_D_RT)

        # Adversarial loss
        DX_loss_M_real = self.loss_function_dict['gan_rt'](pred_labels_M, True).mean()
        DX_loss_M_fake = self.loss_function_dict['gan_tr'](pred_labels_gen_M_TR, False).mean()
        DX_loss_M_fake += self.loss_function_dict['gan_rt'](pred_labels_gen_M_RT, False).mean()
        D_loss = (DX_loss_M_fake + DX_loss_M_real) * 0.5

        log_dict['loss_D'] = D_loss.item()

        return D_loss

    def compute_G_loss(self,tensor_dict, model_dict, log_dict={}):

        pred_labels_gen_M = model_dict['D_M'](tensor_dict['gen_M_RT'])
        GAN_loss_RT = self.loss_function_dict['gan_rt'](pred_labels_gen_M, True).mean()
        GAN_loss_RT = self.loss_weight_dict['gan_rt'] * GAN_loss_RT
        log_dict['loss_' + self.loss_function_dict['gan_rt'].name] = GAN_loss_RT.item()

        pred_labels_gen_M = model_dict['D_M'](tensor_dict['gen_M_TR'])
        GAN_loss_TR = self.loss_function_dict['gan_tr'](pred_labels_gen_M, True).mean()
        GAN_loss_TR = self.loss_weight_dict['gan_tr'] * GAN_loss_TR
        log_dict['loss_' + self.loss_function_dict['gan_tr'].name] = GAN_loss_TR.item()

        REG_loss_RT = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_rt'].name] = REG_loss_RT
        if self.loss_weight_dict['registration_rt'] > 0:
            REG_loss_RT = self.loss_function_dict['registration_rt'](tensor_dict['data_M'], tensor_dict['gen_M_RT'], mask=tensor_dict['mask_M_d'])
            REG_loss_RT = self.loss_weight_dict['registration_rt'] * REG_loss_RT
            log_dict['loss_' + self.loss_function_dict['registration_rt'].name] = REG_loss_RT.item()

        REG_loss_TR = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_tr'].name] = REG_loss_TR
        if self.loss_weight_dict['registration_tr'] > 0:
            REG_loss_TR = self.loss_function_dict['registration_tr'](tensor_dict['data_M'], tensor_dict['gen_M_TR'], mask=tensor_dict['mask_M_d'])
            REG_loss_TR = self.loss_weight_dict['registration_tr'] * REG_loss_TR
            log_dict['loss_' + self.loss_function_dict['registration_tr'].name] = REG_loss_TR.item()

        REGs_loss = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss
        if self.loss_weight_dict['registration_smoothness'] > 0:
            REGs_loss_M = self.loss_function_dict['registration_smoothness'](tensor_dict['velocity_field_M'])
            REGs_loss = self.loss_weight_dict['registration_smoothness'] * REGs_loss_M
            log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss.item()

        # Generator
        GEN_loss = GAN_loss_RT + GAN_loss_TR + REG_loss_TR + REG_loss_RT + REGs_loss
        log_dict['loss_G'] = GEN_loss.item()
        return GEN_loss

    def iterate(self, model_dict, optimizer_dict, generator_train, epoch, **kwargs):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H = data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0#np.random.rand(1)
                flipud = 0#np.random.rand(1)
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr)

            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d
            }

            ##################
            #  Forward pass #
            ##################
            gen_R, flow_field, velocity_field = model_dict['R_M'](data_H, data_M)
            gen_T = model_dict['G_M'](data_H)
            gen_TR = model_dict['R_M'].predict(gen_T, flow_field, svf=False)
            gen_RT = model_dict['G_M'](gen_R)

            tensor_dict['gen_M_TR'] = gen_TR
            tensor_dict['gen_M_RT'] = gen_RT
            tensor_dict['velocity_field_M'] = velocity_field

            ##################
            #  Discriminator #
            ##################
            for p in model_dict['G_M'].parameters():
                p.requires_grad = False  # to avoid computation
            for p in model_dict['R_M'].parameters():
                p.requires_grad = False  # to avoid computation

            for _ in range(self.d_iter):
                optimizer_dict['D_M'].zero_grad()
                D_loss = self.compute_D_loss(tensor_dict, model_dict, log_dict)
                D_loss.backward()
                if self.clip_grad: torch.nn.utils.clip_grad_norm_(model_dict['D_M'].parameters(), self.clip_grad)
                optimizer_dict['D_M'].step()

            for p in model_dict['G_M'].parameters():
                p.requires_grad = True  # to avoid computation
            for p in model_dict['R_M'].parameters():
                p.requires_grad = True  # to avoid computation

            ##############
            #  Generator #
            ##############
            for p in model_dict['D_M'].parameters():
                p.requires_grad = False  # to avoid computation

            optimizer_dict['G_M'].zero_grad()
            optimizer_dict['R_M'].zero_grad()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            optimizer_dict['G_M'].step()
            optimizer_dict['R_M'].step()
            if 'F_M' in optimizer_dict.keys():
                optimizer_dict['F_M'].step()

            for p in model_dict['D_M'].parameters():
                p.requires_grad = True  # to avoid computation

            log_dict['loss'] = log_dict['loss_G'] + log_dict['loss_D']

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def train(self, model_dict, optimizer_dict, generator_train, weakly_flag, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        logs_dict = {}
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            model_dict['G_M'].train()
            model_dict['D_M'].train()
            model_dict['R_M'].train()

            self.iterate(model_dict, optimizer_dict, generator_train, epoch, **kwargs)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)

        print('All done')


class CycleGAN(object):
    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model, clip_grad, p_dict, d_iter=1):
        self.device = device
        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss_G', 'loss_D', 'loss', 'time_duration (s)']

        attach = True if p_dict['STARTING_EPOCH'] > 0 else False
        logger = History(self.log_keys)
        mcheck = ModelCheckpoint(join(p_dict['RESULTS_DIR'], 'checkpoints'), p_dict['SAVE_MODEL_FREQUENCY'])
        training_printer = PrinterCallback()
        training_tocsv = ToCSVCallback(filepath=join(p_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                                       keys=self.log_keys, attach=attach)
        callback_list = [logger, mcheck, training_printer, training_tocsv]

        self.callbacks = callback_list + callbacks

        self.da_model = da_model if da_model is not None else False
        self.clip_grad = clip_grad
        self.d_iter = d_iter
        self.parameter_dict = p_dict

    def compute_D_loss(self, tensor_dict, model_dict, log_dict={}):

        data_M, gen_M = tensor_dict['data_M'], tensor_dict['gen_M']
        data_H, gen_H = tensor_dict['data_H'], tensor_dict['gen_H']
        gen_MD = gen_M.detach()
        gen_HD = gen_H.detach()

        pred_labels_M = model_dict['D_M'](data_M)
        pred_labels_H = model_dict['D_M'](data_H)
        pred_labels_gen_M = model_dict['D_M'](gen_MD)
        pred_labels_gen_H = model_dict['D_M'](gen_HD)

        # Adversarial loss
        DX_loss_M_real = self.loss_function_dict['gan'](pred_labels_M, True).mean()
        DX_loss_M_fake = self.loss_function_dict['gan'](pred_labels_gen_M, False).mean()

        DX_loss_H_real = self.loss_function_dict['gan'](pred_labels_H, True).mean()
        DX_loss_H_fake = self.loss_function_dict['gan'](pred_labels_gen_H, False).mean()

        D_loss = (DX_loss_M_fake + DX_loss_M_real) * 0.5 + (DX_loss_H_fake + DX_loss_H_real) * 0.5

        log_dict['loss_D'] = D_loss.item()

        return D_loss

    def compute_G_loss(self,tensor_dict, model_dict, log_dict={}):

        pred_labels_gen_M = model_dict['D_M'](tensor_dict['gen_M'])
        pred_labels_gen_H = model_dict['D_H'](tensor_dict['gen_H'])

        GAN_loss_M = self.loss_function_dict['gan'](pred_labels_gen_M, True).mean()
        GAN_loss_H = self.loss_function_dict['gan'](pred_labels_gen_H, True).mean()
        GAN_loss = self.loss_weight_dict['gan'] * (GAN_loss_M + GAN_loss_H)

        log_dict['loss_' + self.loss_function_dict['gan'].name] = GAN_loss.item()

        # Registration
        REG_loss = 0.0
        log_dict['loss_' + self.loss_function_dict['registration'].name] = REG_loss
        if self.loss_weight_dict['registration'] > 0:
            REG_loss_M = self.loss_function_dict['registration'](tensor_dict['data_M'], tensor_dict['gen_reg_M'], mask=tensor_dict['mask_M_d'])
            REG_loss = self.loss_weight_dict['registration'] * REG_loss_M
            log_dict['loss_' + self.loss_function_dict['registration'].name] = REG_loss.item()

        REGs_loss = 0.0
        log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss
        if self.loss_weight_dict['registration_smoothness'] > 0:
            REGs_loss_M = self.loss_function_dict['registration_smoothness'](tensor_dict['velocity_field_M'])
            REGs_loss = self.loss_weight_dict['registration_smoothness'] * REGs_loss_M
            log_dict['loss_' + self.loss_function_dict['registration_smoothness'].name] = REGs_loss.item()

        # Cycle
        CYCLE_loss = 0.0
        log_dict['loss_' + self.loss_function_dict['cycle'].name] = CYCLE_loss
        if self.loss_weight_dict['cycle'] > 0:
            CYCLE_loss_M =  self.loss_function_dict['cycle'](tensor_dict['data_M'], tensor_dict['rec_M'])
            CYCLE_loss_H = self.loss_function_dict['cycle'](tensor_dict['data_H'], tensor_dict['rec_H'])
            CYCLE_loss = self.loss_weight_dict['cycle'] * (CYCLE_loss_M + CYCLE_loss_H)
            log_dict['loss_' + self.loss_function_dict['cycle'].name] = CYCLE_loss.item()

        # Generator
        GEN_loss = GAN_loss + REG_loss + REGs_loss + CYCLE_loss
        log_dict['loss_G'] = GEN_loss.item()
        return GEN_loss

    def iterate(self, model_dict, optimizer_dict, generator_train, epoch, **kwargs):

        '''

        '''

        batch_size = generator_train.batch_size
        N = len(generator_train.dataset)
        for batch_idx, data_dict in enumerate(generator_train):
            log_dict = {}

            ###############
            #  Input data #
            ###############

            data_H, data_M = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            mask_H = data_dict['x_flo_init_mask'].to(self.device)
            mask_M_d, mask_M = data_dict['x_ref_mask'].to(self.device), data_dict['x_ref_init_mask'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine = [nlf.to(self.device) for nlf in data_dict['affine']]

            if self.da_model:
                fliplr = 0
                flipud = 0
                data_H = self.da_model.transform(data_H, affine[0], nonlinear_field[0], flipud, fliplr)
                mask_H = self.da_model.transform(mask_H, affine[0], nonlinear_field[0], flipud, fliplr)

                data_M = self.da_model.transform(data_M, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M_d = self.da_model.transform(mask_M_d, affine[1], nonlinear_field[1], flipud, fliplr)
                mask_M = self.da_model.transform(mask_M, affine[1], nonlinear_field[1], flipud, fliplr)

            tensor_dict = {
                'data_H': data_H,
                'data_M': data_M,
                'mask_H': mask_H,
                'mask_M': mask_M,
                'mask_M_d': mask_M_d
            }

            ###################
            #   Forward pass  #
            ###################

            gen_M = model_dict['G_M'](data_H)
            gen_H = model_dict['G_H'](data_M)
            rec_M = model_dict['G_M'](gen_H)
            rec_H = model_dict['G_H'](gen_M)
            gen_reg_M, flow_field_M, velocity_field_M = model_dict['R_M'](gen_M, data_M)

            tensor_dict['gen_M'] = gen_M
            tensor_dict['gen_H'] = gen_H
            tensor_dict['rec_M'] = rec_M
            tensor_dict['rec_H'] = rec_H
            tensor_dict['gen_reg_M'] = gen_reg_M
            tensor_dict['velocity_field_M'] = velocity_field_M

            ##################
            #  Discriminator #
            ##################
            for p in model_dict['D_M'].parameters():
                p.requires_grad = True  # to avoid computation

            for p in model_dict['D_H'].parameters():
                p.requires_grad = True  # to avoid computation

            for p in model_dict['G_M'].parameters():
                p.requires_grad = False  # to avoid computation

            for p in model_dict['G_H'].parameters():
                p.requires_grad = False  # to avoid computation

            for _ in range(self.d_iter):
                optimizer_dict['D_M'].zero_grad()
                optimizer_dict['D_H'].zero_grad()

                D_loss = self.compute_D_loss(tensor_dict, model_dict, log_dict)
                D_loss.backward()

                if self.clip_grad: torch.nn.utils.clip_grad_norm_(model_dict['D_M'].parameters(), self.clip_grad)
                if self.clip_grad: torch.nn.utils.clip_grad_norm_(model_dict['D_H'].parameters(), self.clip_grad)

                optimizer_dict['D_M'].step()
                optimizer_dict['D_H'].step()

            ##############
            #  Generator #
            ##############

            for p in model_dict['D_M'].parameters():
                p.requires_grad = False  # to avoid computation

            for p in model_dict['D_H'].parameters():
                p.requires_grad = False  # to avoid computation

            for p in model_dict['G_M'].parameters():
                p.requires_grad = True  # to avoid computation

            for p in model_dict['G_H'].parameters():
                p.requires_grad = True  # to avoid computation

            optimizer_dict['G_M'].zero_grad()
            optimizer_dict['G_H'].zero_grad()

            G_loss = self.compute_G_loss(tensor_dict, model_dict, log_dict)
            G_loss.backward()

            optimizer_dict['G_M'].step()
            optimizer_dict['G_H'].step()

            log_dict['loss'] = log_dict['loss_G'] + log_dict['loss_D']

            # Callbacks
            for cb in self.callbacks:
                cb.on_step_fi(log_dict, model_dict, epoch, iteration=(batch_idx + 1) * batch_size, N=N)

        return self.callbacks

    def train(self, model_dict, optimizer_dict, generator_train, weakly_flag, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        logs_dict = {}
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            model_dict['G_M'].train()
            model_dict['G_H'].train()
            model_dict['D_M'].train()
            model_dict['D_H'].train()

            self.iterate(model_dict, optimizer_dict, generator_train, epoch, **kwargs)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        plot_results(join(self.parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'), keys=self.log_keys)

        print('All done')

class Registration(object):
    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model, p_dict):
        self.device = device
        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss', 'time_duration (s)']

        attach = True if p_dict['STARTING_EPOCH'] > 0 else False
        logger = History(self.log_keys)
        mcheck = ModelCheckpoint(join(p_dict['RESULTS_DIR'], 'checkpoints'), p_dict['SAVE_MODEL_FREQUENCY'])
        training_printer = PrinterCallback()
        training_tocsv = ToCSVCallback(filepath=join(p_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                                       keys=self.log_keys, attach=attach)
        callback_list = [logger, mcheck, training_printer, training_tocsv]

        self.callbacks = callback_list + callbacks
        self.da_model = da_model if da_model is not None else False
        self.parameter_dict = p_dict

    def iterate_bidir(self, generator, model, optimizer, epoch, mask_flag, **kwargs):

        N = len(generator.dataset)
        rid_epoch_list = []
        total_iter = 0
        for batch_idx, data_dict in enumerate(generator):
            flo_image, ref_image = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine_field = [aff.to(self.device) for aff in data_dict['affine']]
            rid_epoch_list.extend(data_dict['rid'])
            model.zero_grad()

            fliplr = 0
            flipud = 0

            ref_image = self.da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flipud, fliplr)
            flo_image = self.da_model.transform(flo_image, affine_field[1], nonlinear_field[1], flipud, fliplr)

            reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
            flow_image_rev = model.get_flow_field(-v_image)

            if isinstance(model, models.VxmRigidDense):
                ref_image_aff = model.predict_affine(ref_image, flow_image[0])
                reg_ref_image = model.predict(ref_image_aff, flow_image_rev, svf=False)
            else:
                reg_ref_image = model.predict(ref_image, flow_image_rev, svf=False)

            loss_dict = {}
            if mask_flag:
                flo_mask, ref_mask = data_dict['x_flo_mask'].to(self.device), data_dict['x_ref_mask'].to(self.device)
                ref_mask = self.da_model.transform(ref_mask, affine_field[0], nonlinear_field[0], flipud, fliplr, mode='nearest')
                flo_mask = self.da_model.transform(flo_mask, affine_field[1], nonlinear_field[1], flipud, fliplr, mode='nearest')

                loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image, mask=ref_mask)
                loss_dict['registration'] += self.loss_function_dict['registration'](flo_image, reg_ref_image, mask=flo_mask)

            else:
                loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image)
                loss_dict['registration'] += self.loss_function_dict['registration'](flo_image, reg_ref_image)

            loss_dict['registration'] = 0.5 * loss_dict['registration']
            loss_dict['registration_smoothness'] = self.loss_function_dict['registration_smoothness'](v_image)

            for k, v in loss_dict.items():
                loss_dict[k] = self.loss_weight_dict[k] * v

            total_loss = sum([l for l in loss_dict.values()])

            total_loss.backward()
            # plot_grad_flow(model.named_parameters(), save_dir='model_reg')
            optimizer.step()

            log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                        **{'loss': total_loss.item()}
                        }

            total_iter += len(flo_image)
            if batch_idx % kwargs['log_interval'] == 0:  # Logging
                for cb in self.callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

        return self.callbacks

    def iterate(self, generator, model, optimizer, epoch,  mask_flag, **kwargs):

        N = len(generator.dataset)
        rid_epoch_list = []
        total_iter = 0
        for batch_idx, data_dict in enumerate(generator):
            flo_image, ref_image = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine_field = [aff.to(self.device) for aff in data_dict['affine']]
            rid_epoch_list.extend(data_dict['rid'])
            model.zero_grad()

            fliplr = 0
            flipud = 0

            ref_image = self.da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flipud, fliplr)
            flo_image = self.da_model.transform(flo_image, affine_field[1], nonlinear_field[1], flipud, fliplr)

            reg_flo_image, flow_image, v_image = model(flo_image, ref_image)

            loss_dict = {}
            if mask_flag:
                ref_mask = data_dict['x_ref_mask'].to(self.device)
                ref_mask = self.da_model.transform(ref_mask, affine_field[0], nonlinear_field[0], flipud, fliplr, mode='nearest')
                loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image, mask=ref_mask)
            else:
                loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image)
            loss_dict['registration_smoothness'] = self.loss_function_dict['registration_smoothness'](v_image)

            for k, v in loss_dict.items():
                loss_dict[k] = self.loss_weight_dict[k] * v

            total_loss = sum([l for l in loss_dict.values()])

            total_loss.backward()
            # plot_grad_flow(model.named_parameters(), save_dir='model_reg')
            optimizer.step()

            log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                        **{'loss': total_loss.item()}
                        }

            total_iter += len(flo_image)
            if batch_idx % kwargs['log_interval'] == 0:  # Logging
                for cb in self.callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

        return self.callbacks

    def train(self, model, optimizer, generator, bidir_flag, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model)

        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):
            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model, epoch)

            model.train()
            if bidir_flag:
                self.iterate_bidir(generator, model, optimizer, epoch, **kwargs)
            else:
                self.iterate(generator, model, optimizer, epoch, **kwargs)

            epoch_end_time = time.time()
            logs_dict = {}
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model, epoch, optimizer=optimizer)

        for cb in self.callbacks:
            cb.on_train_fi(model)

class WeaklyRegistration(Registration):

    def iterate_bidir(self, generator, model, optimizer, epoch, **kwargs):

        N = len(generator.dataset)
        rid_epoch_list = []
        total_iter = 0
        for batch_idx, data_dict in enumerate(generator):
            flo_image, ref_image = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            flo_mask, ref_mask = data_dict['x_flo_mask'].to(self.device), data_dict['x_ref_mask'].to(self.device)
            flo_labels, ref_labels = data_dict['x_flo_labels'].to(self.device), data_dict['x_ref_labels'].to(self.device)

            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine_field = [aff.to(self.device) for aff in data_dict['affine']]
            rid_epoch_list.extend(data_dict['rid'])
            model.zero_grad()

            fliplr = 0#np.random.rand(1)
            flipud = 0#np.random.rand(1)

            ref_image = self.da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flipud, fliplr)
            ref_mask = self.da_model.transform(ref_mask, affine_field[0], nonlinear_field[0], flipud, fliplr)
            ref_labels = self.da_model.transform(ref_labels, affine_field[0], nonlinear_field[0], flipud, fliplr)

            flo_image = self.da_model.transform(flo_image, affine_field[1], nonlinear_field[1], flipud, fliplr)
            flo_mask = self.da_model.transform(flo_mask, affine_field[1], nonlinear_field[1], flipud, fliplr)
            flo_labels = self.da_model.transform(flo_labels, affine_field[1], nonlinear_field[1], flipud, fliplr)

            reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
            flow_image_rev = model.get_flow_field(-v_image)

            if isinstance(model, models.VxmRigidDense):
                reg_flo_labels_aff = model.predict_affine(flo_labels, flow_image[0], inverse=False)
                reg_flo_labels = model.predict(reg_flo_labels_aff, flow_image[1], svf=False)

                ref_image_aff = model.predict_affine(ref_image, flow_image[0], inverse=False)
                reg_ref_image = model.predict(ref_image_aff, flow_image_rev, svf=False)
                ref_labels_aff = model.predict_affine(ref_image, flow_image[0], inverse=False)
                reg_ref_labels = model.predict_affine(ref_labels_aff, flow_image_rev, svf=False)

            else:
                reg_ref_image = model.predict(ref_image, flow_image_rev, svf=False)
                reg_ref_labels = model.predict(ref_labels, flow_image_rev, svf=False)
                reg_flo_labels = model.predict(flo_labels, flow_image, svf=False)

            loss_dict = {}
            loss_dict['registration'] = self.loss_function_dict['registration'](reg_flo_image, ref_image, mask=ref_mask)
            loss_dict['registration'] += self.loss_function_dict['registration'](reg_ref_image, flo_image, mask=flo_mask)
            loss_dict['registration'] = 0.5 * loss_dict['registration']

            loss_dict['registration_labels'] = self.loss_function_dict['registration_labels'](reg_flo_labels, ref_labels)
            loss_dict['registration_labels'] += self.loss_function_dict['registration_labels'](reg_ref_labels, flo_labels)
            loss_dict['registration_labels'] = 0.5 * loss_dict['registration_labels']

            loss_dict['registration_smoothness'] = self.loss_function_dict['registration_smoothness'](v_image)

            for k, v in loss_dict.items():
                loss_dict[k] = self.loss_weight_dict[k] * v

            total_loss = sum([l for l in loss_dict.values()])

            total_loss.backward()
            # plot_grad_flow(model.named_parameters(), save_dir='model_reg')
            optimizer.step()

            log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                        **{'loss': total_loss.item()}
                        }

            total_iter += len(flo_image)
            if batch_idx % kwargs['log_interval'] == 0:  # Logging
                for cb in self.callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

        return self.callbacks

    def iterate(self, generator, model, optimizer, epoch, **kwargs):

        N = len(generator.dataset)
        rid_epoch_list = []
        total_iter = 0
        for batch_idx, data_dict in enumerate(generator):
            flo_image, ref_image = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
            ref_mask = data_dict['x_ref_mask'].to(self.device)
            flo_labels, ref_labels = data_dict['x_flo_labels'].to(self.device), data_dict['x_ref_labels'].to(self.device)

            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine_field = [aff.to(self.device) for aff in data_dict['affine']]
            rid_epoch_list.extend(data_dict['rid'])
            model.zero_grad()

            fliplr = 0#np.random.rand(1)
            flipud = 0#np.random.rand(1)

            ref_image = self.da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flipud, fliplr)
            ref_mask = self.da_model.transform(ref_mask, affine_field[0], nonlinear_field[0], flipud, fliplr)
            ref_labels = self.da_model.transform(ref_labels, affine_field[0], nonlinear_field[0], flipud, fliplr, mode='nearest')

            flo_image = self.da_model.transform(flo_image, affine_field[1], nonlinear_field[1], flipud, fliplr)
            flo_labels = self.da_model.transform(flo_labels, affine_field[1], nonlinear_field[1], flipud, fliplr, mode='nearest')

            reg_flo_image, flow_image, v_image = model(flo_image, ref_image)

            if isinstance(model, models.VxmRigidDense):
                reg_flo_labels_aff = model.predict_affine(flo_labels, flow_image[0], inverse=False)
                reg_flo_labels = model.predict(reg_flo_labels_aff, flow_image[1], svf=False)

            else:
                reg_flo_labels = model.predict(flo_labels, flow_image, svf=False)

            loss_dict = {}
            loss_dict['registration'] = self.loss_function_dict['registration'](reg_flo_image, ref_image, mask=ref_mask)
            loss_dict['registration_labels'] = self.loss_function_dict['registration_labels'](reg_flo_labels, ref_labels)
            loss_dict['registration_smoothness'] = self.loss_function_dict['registration_smoothness'](v_image)

            for k, v in loss_dict.items():
                loss_dict[k] = self.loss_weight_dict[k] * v

            total_loss = sum([l for l in loss_dict.values()])

            total_loss.backward()
            # plot_grad_flow(model.named_parameters(), save_dir='model_reg')
            optimizer.step()

            log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                        **{'loss': total_loss.item()}
                        }

            total_iter += len(flo_image)
            if batch_idx % kwargs['log_interval'] == 0:  # Logging
                for cb in self.callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

        return self.callbacks

class Segmentation(object):
    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, da_model, p_dict, bidir=True):
        self.device = device
        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] + ['loss', 'time_duration (s)']

        attach = True if p_dict['STARTING_EPOCH'] > 0 else False
        logger = History(self.log_keys)
        mcheck = ModelCheckpoint(join(p_dict['RESULTS_DIR'], 'checkpoints'), p_dict['SAVE_MODEL_FREQUENCY'])
        training_printer = PrinterCallback()
        training_tocsv = ToCSVCallback(filepath=join(p_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                                       keys=self.log_keys, attach=attach)
        callback_list = [logger, mcheck, training_printer, training_tocsv]

        self.callbacks = callback_list + callbacks
        self.da_model = da_model if da_model is not None else False
        self.parameter_dict = p_dict
        self.bidir = bidir

    def iterate(self, generator, model, optimizer, epoch, **kwargs):

        N = len(generator.dataset)
        rid_epoch_list = []
        total_iter = 0
        for batch_idx, data_dict in enumerate(generator):
            ref_image =  data_dict['x_ref'].to(self.device)
            ref_labels = data_dict['x_ref_labels'].to(self.device)
            nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            affine_field = [aff.to(self.device) for aff in data_dict['affine']]
            rid_epoch_list.extend(data_dict['rid'])
            model.zero_grad()

            fliplr = 0
            flipud = 0

            ref_image = self.da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flipud, fliplr)
            ref_labels = self.da_model.transform(ref_labels, affine_field[0], nonlinear_field[0], flipud, fliplr, mode='nearest')

            ref_prob_seg = model(ref_image)

            # if epoch > 0:
            #     pdb.set_trace()
            #     import nibabel as nib
            #     img = nib.Nifti1Image(np.argmax(ref_labels[0].detach().cpu().numpy(), axis=0).astype('uint16'), np.eye(4))
            #     nib.save(img, 'labels_H.nii.gz')
            #     img = nib.Nifti1Image(np.argmax(ref_prob_seg[0].detach().cpu().numpy(), axis=0).astype('uint16'), np.eye(4))
            #     nib.save(img, 'labels_M.nii.gz')

            loss_dict = {}
            loss_dict['segmentation'] = self.loss_function_dict['segmentation'](ref_labels, ref_prob_seg, classes_compute=list(range(1,ref_labels.shape[1])))
            for k, v in loss_dict.items():
                loss_dict[k] = self.loss_weight_dict[k] * v

            total_loss = sum([l for l in loss_dict.values()])

            total_loss.backward()
            optimizer.step()

            log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                        **{'loss': total_loss.item()}
                        }

            total_iter += len(ref_image)
            if batch_idx % kwargs['log_interval'] == 0:  # Logging
                for cb in self.callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

        return self.callbacks

    def train(self, model, optimizer, generator, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model)

        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):
            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model, epoch)

            model.train()
            self.iterate(generator, model, optimizer, epoch, **kwargs)

            epoch_end_time = time.time()
            logs_dict = {}
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model, epoch, optimizer=optimizer)

        for cb in self.callbacks:
            cb.on_train_fi(model)

