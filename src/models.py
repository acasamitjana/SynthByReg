# py
import os
import functools

# third party imports
import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

#project imports
from src.layers import Upsample, Downsample, Normalize, ResnetBlock, ConvBlock2D, ConvBlock3D, SpatialTransformer, VecInt, \
    RescaleTransform, ResnetBlock3D, Downsample3D, Upsample3D, SpatialTransformerAffine
from src.utils.tensor_utils import init_net



class BaseModel(nn.Module):
    pass

########################################
##   Discriminators
########################################
class NLayerDiscriminator(BaseModel):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False,
                 init_type='xavier', init_gain=0.02, device='cpu'):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        model = nn.Sequential(*sequence)

        self.model = init_net(model, init_type, init_gain, device)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class NLayerDiscriminator3D(BaseModel):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False,
                 init_type='xavier', init_gain=0.02, device='cpu'):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample3D(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample3D(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        model = nn.Sequential(*sequence)

        self.model = init_net(model, init_type, init_gain, device)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PatchSampleF(BaseModel):
    def __init__(self, use_mlp=False, nc=256, init_type='normal', init_gain=0.02, device='cpu'):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.device = device

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc, bias=True), nn.ReLU(True), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, device=self.device, init_bias=0.0001)
        self.mlp_init = True

    # def forward(self, feats, num_patches=64, patch_ids=None):
    #     return_ids = []
    #     return_feats = []
    #     if self.use_mlp and not self.mlp_init:
    #         self.create_mlp(feats)
    #     for feat_id, feat in enumerate(feats):
    #         B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
    #         feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
    #         if num_patches > 0:
    #             if patch_ids is not None:
    #                 patch_id = patch_ids[feat_id]
    #             else:
    #                 patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
    #                 patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
    #             x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
    #         else:
    #             x_sample = feat_reshape
    #             patch_id = []
    #         if self.use_mlp:
    #             mlp = getattr(self, 'mlp_%d' % feat_id)
    #             x_sample = mlp(x_sample)
    #         return_ids.append(patch_id)
    #         x_sample = self.l2norm(x_sample)
    #
    #         if num_patches == 0:
    #             x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
    #         return_feats.append(x_sample)
    #
    #     return return_feats, return_ids

    def forward(self, feats, num_patches=64, patch_ids=None, mask_sampling=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        for feat_id, feat in enumerate(feats):

            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            if B>1 and mask_sampling is not None:
                raise ValueError('Mask sampling is only available for batch_size=1')
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    if mask_sampling is None:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=self.device)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    else:
                        mask = mask_sampling[feat_id]
                        mask_reshape = mask.permute(0, 2, 3, 1).flatten(1, 2)
                        idx = torch.nonzero(mask_reshape[0,:,0])[:, 0]
                        idx_perm = torch.randperm(idx.shape[0], device=self.device)
                        idx_perm = idx_perm[:int(min(num_patches, idx_perm.shape[0]))]
                        patch_id = idx[idx_perm]
                        # patch_id = torch.randperm(feat_reshape.shape[1], device=self.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # shape: patches X channels
            else:
                x_sample = feat_reshape
                patch_id = []

            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                # print('MLP')
                # print(x_sample.mean())
                # print([a + '_' + str(b.mean()) for a,b in mlp.named_parameters()])
                x_sample = mlp(x_sample)
                # print(x_sample.mean())

            return_ids.append(patch_id)

            x_sample = self.l2norm(x_sample)# channelwise normalization by the l2norm

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)

        return return_feats, return_ids

class PatchSampleF3D(PatchSampleF):

    def forward(self, feats, num_patches=64, patch_ids=None, mask_sampling=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W, D = feat.shape[0], feat.shape[2], feat.shape[3], feat.shape[4]
            if B>1 and mask_sampling is not None:
                raise ValueError('Mask sampling is only available for batch_size=1')
            feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    if mask_sampling is None:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=self.device)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    else:
                        mask = mask_sampling[feat_id]
                        mask_reshape = mask.permute(0, 2, 3, 4, 1).flatten(1, 3)
                        idx = torch.nonzero(mask_reshape[0,:,0])[:, 0]
                        idx_perm = torch.randperm(idx.shape[0], device=self.device)
                        idx_perm = idx_perm[:int(min(num_patches, idx_perm.shape[0]))]
                        patch_id = idx[idx_perm]
                        # patch_id = torch.randperm(feat_reshape.shape[1], device=self.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # shape: patches X channels
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                # print('MLP')
                # print(x_sample.mean())
                # print([a + '_' + str(b.mean()) for a,b in mlp.named_parameters()])
                x_sample = mlp(x_sample)
                # print(x_sample.mean())

            return_ids.append(patch_id)

            x_sample = self.l2norm(x_sample) # channelwise normalization by the l2norm

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)

        return return_feats, return_ids

########################################
##   UNet like
########################################

class ResnetGenerator(BaseModel):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None,
                 n_downsampling=2,init_type='xavier', init_gain=0.02, device='cpu', tanh=True):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()] if tanh else [nn.Sigmoid()]

        model = nn.Sequential(*model)
        self.model = init_net(model, init_type, init_gain, device)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

class ResnetGenerator3D(BaseModel):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None,
                 n_downsampling=2,init_type='xavier', init_gain=0.02, device='cpu', tanh=True):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator3D, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.ConstantPad3d(3, -1.0),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample3D(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock3D(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample3D(ngf * mult),
                          nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ConstantPad3d(3, -1)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()] if tanh else [nn.Sigmoid()]

        model = nn.Sequential(*model)
        self.model = init_net(model, init_type, init_gain, device)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

class SegUnet(BaseModel):
    """
    Voxelmorph Unet. For more information see voxelmorph.net
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, num_classes, nb_features=None, nb_levels=None, feat_mult=1, activation='lrelu'):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = self._default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        nlayers_uparm = len(self.enc_nf)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            if ndims == 2:
                self.downarm.append(ConvBlock2D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.downarm.append(ConvBlock3D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:nlayers_uparm]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            if ndims == 2:
                self.uparm.append(ConvBlock2D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.uparm.append(ConvBlock3D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))

            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf + 1
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            if ndims == 2:
                self.extras.append(ConvBlock2D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.extras.append(ConvBlock3D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        self.extras.append(ConvBlock3D(prev_nf, num_classes, stride=1, activation='softmax', norm_layer='none', padding=1))

    def _default_unet_features(self):
        nb_features = [
            [16, 32, 32, 32],  # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
        return nb_features

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class VxmUnet(BaseModel):
    """
    Voxelmorph Unet. For more information see voxelmorph.net
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, activation='lrelu',
                 cpoints_level=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = self._default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        nlayers_uparm = len(self.enc_nf) - int(np.log2(cpoints_level))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            if ndims == 2:
                self.downarm.append(ConvBlock2D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.downarm.append(ConvBlock3D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:nlayers_uparm]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            if ndims == 2:
                self.uparm.append(ConvBlock2D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.uparm.append(ConvBlock3D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))

            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf + 2 if cpoints_level == 1 else prev_nf + enc_history[nlayers_uparm]
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            if ndims == 2:
                self.extras.append(ConvBlock2D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.extras.append(ConvBlock3D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

    def _default_unet_features(self):
        nb_features = [
            [16, 32, 32, 32],  # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
        return nb_features

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class VxmRigidParams(BaseModel):
    """
        Voxelmorph Unet. For more information see voxelmorph.net
        A unet architecture. Layer features can be specified directly as a list of encoder and decoder
        features or as a single integer along with a number of unet levels. The default network features
        per layer (when no options are specified) are:
            encoder: [16, 32, 32, 32]
            decoder: [32, 32, 32, 32, 32, 16, 16]
        """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, activation='lrelu', device='cpu'):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        self.device = device
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = self._default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            if ndims == 2:
                self.downarm.append(
                    ConvBlock2D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.downarm.append(
                    ConvBlock3D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        if ndims == 2:
            self.linear = nn.Linear(prev_nf, 3) #3 angle + 3 tx
        else:
            self.linear = nn.Linear(prev_nf, 6) #angle + 2 tx


    def _default_unet_features(self):
        return [16, 32, 32, 32]

    def forward(self, x):
        ndims = len(x.shape) - 2
        batch_size = x.shape[0]
        cr = torch.tensor([x.shape[i+2]/2 for i in range(ndims)])
        # get encoder activations
        x_enc = x
        for layer in self.downarm:
            x_enc = layer(x_enc)

        x_enc = torch.mean(x_enc, dim=[i + 2 for i in range(ndims)])
        lin_params = self.linear(x_enc)
        if ndims == 2:
            T1 = torch.eye(3).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            T2 = torch.eye(3).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            T3 = torch.eye(3).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            T4 = torch.eye(3).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)

            T1[:, 2, 2] = -cr.unsqueeze(0).repeat(batch_size,1)

            T2[0, 0] = torch.cos(lin_params[:,0]*np.pi/180)
            T2[:, 0, 1] = -torch.sin(lin_params[:,0]*np.pi/180)
            T2[:, 1, 0] = torch.sin(lin_params[:,0]*np.pi/180)
            T2[:, 1, 1] = torch.cos(lin_params[:,0]*np.pi/180)

            T3[:2, 2] = cr[0].unsqueeze(0).repeat(batch_size, 1)

            T4[:, :2, 2] = lin_params[:,1:]

            T = torch.chain_matmul(T4, T3, T2, T1)


        else:
            T=[]
            blin_params = torch.unbind(lin_params, dim=0)
            for blp in blin_params:
                T1 = torch.eye(4)
                T2 = torch.eye(4)
                T3 = torch.eye(4)
                T4 = torch.eye(4)
                T5 = torch.eye(4)
                T6 = torch.eye(4)

                T1[:3, 3] = -cr

                T2[1, 1] = torch.cos(blp[0]*np.pi/180)
                T2[1, 2] = -torch.sin(blp[0]*np.pi/180)
                T2[2, 1] = torch.sin(blp[0]*np.pi/180)
                T2[2, 2] = torch.cos(blp[0]*np.pi/180)

                T3[0, 0] = torch.cos(blp[1]*np.pi/180)
                T3[0, 2] = torch.sin(blp[1]*np.pi/180)
                T3[2, 0] = -torch.sin(blp[1]*np.pi/180)
                T3[2, 2] = torch.cos(blp[1]*np.pi/180)

                T4[0, 0] = torch.cos(blp[2]*np.pi/180)
                T4[0, 1] = -torch.sin(blp[2]*np.pi/180)
                T4[1, 0] = torch.sin(blp[2]*np.pi/180)
                T4[1, 1] = torch.cos(blp[2]*np.pi/180)

                T5[:3, 3] = cr

                T6[:3, 3] = blp[3:]

                T.append(torch.chain_matmul(T6, T5, T4, T3, T2, T1))

            T = torch.stack(T).to(self.device)
            # T1 = torch.eye(4).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            # T2 = torch.eye(4).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            # T3 = torch.eye(4).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            # T4 = torch.eye(4).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            # T5 = torch.eye(4).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            # T6 = torch.eye(4).unsqueeze(2).repeat(1,1,batch_size).transpose(2,0)
            #
            # T1[:, :3, 3] = -cr.unsqueeze(0).repeat(batch_size,1)
            #
            # T2[:, 1, 1] = torch.cos(lin_params[:,0]*np.pi/180)
            # T2[:,1, 2] = -torch.sin(lin_params[:,0]*np.pi/180)
            # T2[:,2, 1] = torch.sin(lin_params[:,0]*np.pi/180)
            # T2[:,2, 2] = torch.cos(lin_params[:,0]*np.pi/180)
            #
            # T3[:,0, 0] = torch.cos(lin_params[:,1]*np.pi/180)
            # T3[:,0, 2] = torch.sin(lin_params[:,1]*np.pi/180)
            # T3[:,2, 0] = -torch.sin(lin_params[:,1]*np.pi/180)
            # T3[:,2, 2] = torch.cos(lin_params[:,1]*np.pi/180)
            #
            # T4[:,0, 0] = torch.cos(lin_params[:,2]*np.pi/180)
            # T4[:,0, 1] = -torch.sin(lin_params[:,2]*np.pi/180)
            # T4[:,1, 0] = torch.sin(lin_params[:,2]*np.pi/180)
            # T4[:,1, 1] = torch.cos(lin_params[:,2]*np.pi/180)
            #
            # T5[:, :3, 3] = cr.unsqueeze(0).repeat(batch_size,1)
            #
            # T6[:, :3, 3] = lin_params[:,3:]

            # T = torch.chain_matmul(T6, T5, T4, T3, T2, T1)

        return T

########################################
##   Registration
########################################

class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        use_probs=False,
        gaussian_filter_flag=True,
        device='cpu'):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        unet_model = VxmUnet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            cpoints_level=int_downsize
        )
        self.unet_model = init_net(unet_model)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape), requires_grad=True)
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape), requires_grad=True)

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        # self.resize = RescaleTransform(inshape, factor=1 / int_downsize, gaussian_filter_flag=gaussian_filter_flag) if resize else None
        self.resize = None
        self.fullsize = RescaleTransform(inshape, factor=int_downsize) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        if self.resize:
            flow_field = self.resize(flow_field)

        preint_flow = flow_field

        # integrate to produce diffeomorphic warp
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        # warp image with flow field
        y_source = self.transformer(source, flow_field)

        # return non-integrated flow field if training
        if not registration:
            return y_source, flow_field, preint_flow
        else:
            return y_source, flow_field

    def predict(self, image, flow, svf=True, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field

class VxmRigidDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        use_probs=False,
        device='cpu'):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core encoder model
        rigid_model = VxmRigidParams(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            device=device
        )
        self.rigid_model = init_net(rigid_model)

        # configure core unet model
        unet_model = VxmUnet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            cpoints_level=int_downsize
        )
        self.unet_model = init_net(unet_model)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape), requires_grad=True)
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape), requires_grad=True)

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        # self.resize = RescaleTransform(inshape, factor=1 / int_downsize, gaussian_filter_flag=gaussian_filter_flag) if resize else None
        self.resize = None
        self.fullsize = RescaleTransform(inshape, factor=int_downsize) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        self.transformer_affine = SpatialTransformerAffine(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        T = self.rigid_model(x)
        x_aff = self.transformer_affine(source, T)

        # concatenate inputs and propagate unet
        x = torch.cat([x_aff, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        if self.resize:
            flow_field = self.resize(flow_field)

        preint_flow = flow_field

        # integrate to produce diffeomorphic warp
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        # warp image with flow field
        y_source = self.transformer(source, flow_field)

        # return non-integrated flow field if training
        if not registration:
            return y_source, [T, flow_field], preint_flow
        else:
            return y_source, [T, flow_field]

    def predict_affine(self, image, T, inverse=False, **kwargs):
        if inverse:
            T = torch.inverse(T)

        return self.transformer_affine(image, T)

    def predict(self, image, flow, svf=False, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field

