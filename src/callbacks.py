from os.path import join, exists
import csv

import numpy as np
import torch


class Callback(object):

    def on_train_init(self, model, **kwargs):
        pass

    def on_train_fi(self, model, **kwargs):
        pass

    def on_epoch_init(self, model, epoch, **kwargs):
        pass

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_init(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_fi(self, logs_dict, model, epoch,**kwargs):
        pass


class History(Callback):

    def __init__(self, keys=None):
        self.logs = {}

        if keys is None:
            self.keys = []
        else:
            self.keys = keys

    def on_train_init(self, model, **kwargs):
        self.logs['Train'] = {}
        self.logs['Validation'] = {}


    def on_epoch_init(self, model, epoch, **kwargs):
        self.logs['Train'][epoch] = {}
        self.logs['Validation'][epoch] = {}


        for k in self.keys:
            self.logs['Train'][epoch][k] = []


    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        for k,v in logs_dict.items():
            self.logs['Train'][epoch][k].append(v)


    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        for k,v in logs_dict.items():
            self.logs['Validation'][epoch][k]=v


class ModelCheckpoint(Callback):

    def __init__(self, dirpath, save_model_frequency):

        self.dirpath = dirpath
        self.save_model_frequency = save_model_frequency

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):

        optimizer = kwargs['optimizer']
        checkpoint = {
            'epoch': epoch + 1,
        }
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        if isinstance(optimizer, dict):
            for optimizer_name, optimizer_instance in optimizer.items():
                checkpoint['optimizer_' + optimizer_name] = optimizer_instance.state_dict()
        else:
            checkpoint['optimizer'] = optimizer.state_dict()

        filepath = join(self.dirpath, 'model_checkpoint.LAST.pth')
        torch.save(checkpoint, filepath)
        if np.mod(epoch, self.save_model_frequency) == 0:
            filepath = join(self.dirpath, 'model_checkpoint.' + str(epoch) + '.pth')
            torch.save(checkpoint, filepath)


    def on_train_fi(self, model, **kwargs):
        checkpoint = {}
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        filepath = join(self.dirpath, 'model_checkpoint.FI.pth')
        torch.save(checkpoint, filepath)


class PrinterCallback(Callback):

    def __init__(self, keys=None):
        self.keys = keys
        self.logs = {}

    def on_train_init(self, model, **kwargs):
        print('######################################')
        print('########## Training started ##########')
        print('######################################')
        print('\n')

    def on_epoch_init(self, model, epoch, **kwargs):
        print('------------- Epoch: ' + str(epoch))

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        to_print = 'Iteration: (' + str(kwargs['iteration']) + '/' + str(kwargs['N']) + '). ' + \
                   ', '.join([k + ': ' + str(round(v, 3)) for k, v in logs_dict.items()])
        print(to_print)
    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        to_print = 'Epoch summary: ' + ','.join([k + ': ' + str(round(v, 3)) for k, v in logs_dict.items()])
        print(to_print)


    def on_train_fi(self, model, **kwargs):
        print('#######################################')
        print('########## Training finished ##########')
        print('#######################################')


class ToCSVCallback(Callback):

    def __init__(self, filepath, keys, attach=False):
        mode = 'a' if attach else 'w'
        fieldnames = ['Phase','epoch','iteration'] + keys
        write_header = True
        if exists(filepath) and attach:
            write_header = False
        csvfile = open(filepath, mode)
        self.csvwriter = csv.DictWriter(csvfile, fieldnames)

        if write_header:
            self.csvwriter.writeheader()



    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        write_dict = {**{'Phase': 'Train', 'epoch':epoch, 'iteration':kwargs['iteration']}, **logs_dict}
        self.csvwriter.writerow(write_dict)

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        write_dict = {**{'Phase': 'Validation', 'epoch': epoch}, **logs_dict}
        self.csvwriter.writerow(write_dict)


class LRDecay(Callback):

    def __init__(self, optimizer, n_iter_start, n_iter_finish, lr_fi=0.75):
        self.optimizer = optimizer
        self.n_iter_start = n_iter_start
        self.n_iter_finish = n_iter_finish
        self.optimizer = optimizer
        self.lr_init = optimizer.param_groups[0]['lr']
        self.lr_fi = lr_fi

    def on_train_init(self, model, **kwargs):
        init_epoch = kwargs['starting_epoch'] if 'starting_epoch' in kwargs.keys() else 0
        if init_epoch >= self.n_iter_start:
            updated_lr = (1 - self.lr_fi * (init_epoch - self.n_iter_start)/(self.n_iter_finish-self.n_iter_start)) * self.lr_init
            updated_lr = max(updated_lr, self.lr_init*self.lr_fi)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = updated_lr

    def on_epoch_init(self, model, epoch, **kwargs):
        if epoch >= self.n_iter_start:
            updated_lr = (1 - self.lr_fi * (epoch - self.n_iter_start)/(self.n_iter_finish-self.n_iter_start)) * self.lr_init
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = updated_lr



