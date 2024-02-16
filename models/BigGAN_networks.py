# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import functools

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from util.augmentations import ProgressiveWordCrop, CycleWordCrop, StaticWordCrop, RandomWordCrop
from . import BigGAN_layers as layers
from .networks import init_weights
import torchvision
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.

from models.blocks import Conv2dBlock, ResBlocks


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', input_nc=3, ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[63] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [input_nc] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    arch[129] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[33] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[31] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[16] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}

    arch[17] = {'in_channels': [input_nc] + [ch * item for item in [1, 4]],
                 'out_channels': [item * ch for item in [1, 4, 8]],
                 'downsample': [True] * 3,
                 'resolution': [16, 8, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}


    arch[20] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, resolution, D_ch=64, D_wide=True, D_kernel_size=3, D_attn='64',
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-8, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='N02', skip_init=False, D_param='SN', gpu_ids=[0],bn_linear='SN', input_nc=1, one_hot=False, crop_size: list = None, **kwargs):

        super(Discriminator, self).__init__()
        self.crop = crop_size is not None and len(crop_size) > 0

        use_padding = False

        if self.crop:
            w_crop = StaticWordCrop(crop_size[0], use_padding=use_padding) if len(crop_size) == 1 else RandomWordCrop(crop_size[0], crop_size[1], use_padding=use_padding)

            self.augmenter = w_crop

        self.name = 'D'
        # gpu_ids
        self.gpu_ids = gpu_ids
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear=='SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        # Initialize weights
        if not skip_init:
            self = init_weights(self, D_init)

    def update_parameters(self, epoch: int):
        if self.crop:
            self.augmenter.update(epoch)

    def forward(self, x, y=None, **kwargs):
        # Stick x into h for cleaner for loops without flow control
        if self.crop and random.uniform(0.0, 1.0) < 0.33:
            x = self.augmenter(x)

        #imgs = [np.squeeze((img.detach().cpu().numpy() + 1.0) / 2.0) for img in x]
        #imgs = (np.vstack(imgs) * 255.0).astype(np.uint8)
        #cv2.imwrite(f"saved_images/debug/{random.randint(0, 1000)}.jpg", imgs)

        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)

        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        out = self.linear(h)

        return out

    def return_features(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        block_output = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                block_output.append(h)
        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output


class WDiscriminator(nn.Module):

    def __init__(self, resolution, n_classes, output_dim, D_ch=64, D_wide=True, D_kernel_size=3, D_attn='64',
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-8, D_mixed_precision=False, D_fp16=False,
                 D_init='N02', skip_init=False, D_param='SN', gpu_ids=[0],bn_linear='SN', input_nc=1, one_hot=False):
        super(WDiscriminator, self).__init__()

        self.name = 'D'
        # gpu_ids
        self.gpu_ids = gpu_ids
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear == 'SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
        self.cross_entropy = nn.CrossEntropyLoss()
        # Initialize weights
        if not skip_init:
            self = init_weights(self, D_init)

    def update_parameters(self, epoch: int):
        pass

    def forward(self, x, y=None, **kwargs):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])

        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        #if y is not None:
        loss = self.cross_entropy(out, y.long())
        return loss

    def return_features(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        block_output = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                block_output.append(h)
        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output


class Encoder(Discriminator):
    def __init__(self, opt, output_dim, **kwargs):
        super(Encoder, self).__init__(**vars(opt))
        self.output_layer = nn.Sequential(self.activation,
                                          nn.Conv2d(self.arch['out_channels'][-1], output_dim, kernel_size=(4,2), padding=0, stride=2))

    def forward(self, x):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        out = self.output_layer(h)
        return out
