'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Ossicles 6D pose estimation
@file: dataset.py
@time: 2023-07-03 17:37
@desc: Build the network structure
'''

import torch
import torch.nn as nn
torch.cuda.empty_cache() if torch.cuda.is_available() else ""

class UNet(nn.Module):
    def __init__(self, in_channel=3):

        super(UNet, self).__init__()

        self.in_channel = in_channel
        self.pool0 = nn.MaxPool2d(2)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)

        # add the drop out function to prevent overfitting
        # self.dropout1 = nn.Dropout(0.25)

        self.ec0 = self.encoder_layer(self.in_channel, 32, batchnorm=False)
        self.ec1 = self.encoder_layer(32, 64, batchnorm=False)
        self.ec2 = self.encoder_layer(64, 64)
        self.ec3 = self.encoder_layer(64, 128)
        self.ec4 = self.encoder_layer(128, 128)
        self.ec5 = self.encoder_layer(128, 256)
        self.ec6 = self.encoder_layer(256, 256)
        self.ec7 = self.encoder_layer(256, 512)

        self.dc9 = self.decoder_layer(512, 512, stride=2) # for shape (64, 64), (128, 128), ...
        self.dc8 = self.decoder_layer(256 + 512, 256)
        self.dc7 = self.decoder_layer(256, 256)
        self.dc6 = self.decoder_layer(256, 256, stride=2, padding=2) 
        self.dc5 = self.decoder_layer(128 + 256, 128, stride=1)
        self.dc4 = self.decoder_layer(128, 128, stride=1)
        self.dc3 = self.decoder_layer(128, 128, stride=2, padding=2)
        self.dc2 = self.decoder_layer(64 + 128, 64, stride=1, padding=1)
        self.dc1 = self.decoder_layer(64, 64, stride=1, padding=1)

        # keep the heads seperate and use the Sigmoid activation function
        self.dc_cmap = self.decoder_layer(64, 1, kernel_size=3, stride=1, activation=nn.Sigmoid())      
        self.dc_u = self.decoder_layer(64, 1, kernel_size=3, stride=1, activation=nn.Sigmoid())
        self.dc_v = self.decoder_layer(64, 1, kernel_size=3, stride=1, activation=nn.Sigmoid())
        
    @staticmethod
    def encoder_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=True, activation=nn.ReLU()):
        if batchnorm:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                                nn.BatchNorm2d(out_channels, track_running_stats=False),
                                activation)
        # it is common to omit BN in the first down-sampling block (i.e., the first convolutional layer and max pooling layer) to avoid over-regularization of the activations
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                                activation)
        return layer
    
    @staticmethod
    def decoder_layer(in_channels, out_channels, kernel_size=2, stride=1, padding=0, output_padding=0, dilation=1, bias=True, activation=nn.ReLU()):
        layer = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, bias=bias),
                            activation)
        return layer

    def down(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)

        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2
        
        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4
        
        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        # e7 = self.dropout1(e7)
        
        return e7, syn0, syn1, syn2
    
    def up(self, e7, syn0, syn1, syn2):
        temp = self.dc9(e7)
        d9 = torch.cat((temp, syn2),1)
        del temp

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8
        
        temp = self.dc6(d7)
        d6 = torch.cat((temp, syn1),1)
        del d7, temp

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5
        
        temp = self.dc3(d4)
        d3 = torch.cat((temp, syn0),1)
        del d4, temp

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        return d1

    def forward(self, x):
        # Encoder
        e7, syn0, syn1, syn2 = self.down(x)

        # Confidence map Decoder, UV Decoders and heads
        d1_cmap = self.up(e7, syn0, syn1, syn2)
        d1_u = self.up(e7, syn0, syn1, syn2)
        d1_v = self.up(e7, syn0, syn1, syn2)
        d_cmap = self.dc_cmap(d1_cmap)
        d_u = self.dc_u(d1_u)
        d_v = self.dc_v(d1_v)
        d_uv = torch.cat((d_u, d_v), dim=1).float()
        del d1_cmap, d1_u, d1_v, d_u, d_v, e7, syn0, syn1, syn2

        return d_uv
