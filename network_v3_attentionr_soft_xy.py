import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

def conv_S(in_planes,out_planes,stride=1,padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=1,
                     padding=padding,bias=False)

def conv_T(in_planes,out_planes,stride=1,padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=1,
                     padding=padding,bias=False)

def binarize(x):
    return (x>0.5).float()

class _P3Dconv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_mode = 'a'):
        super(_P3Dconv,self).__init__()
        self.conv_mode = conv_mode.lower()
        if self.conv_mode == '2d':
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False)
        elif self.conv_mode == 'a':
            mid = (in_channels+out_channels)//2
            self.conS = conv_S(in_channels,mid,padding=(0,1,1))
            self.conT = conv_T(mid,out_channels,padding=(1,0,0))
        elif self.conv_mode == 'b':
            self.conS = conv_S(in_channels,out_channels,padding=(0,1,1))
            self.conT = conv_T(in_channels,out_channels,padding=(1,0,0))
        elif self.conv_mode == 'c':
            self.conS = conv_S(in_channels,out_channels,padding=(0,1,1))
            self.conT = conv_T(out_channels,out_channels,padding=(1,0,0))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        if self.conv_mode == '2d':
            temp = self.conv2d(x)
        elif self.conv_mode == 'a':
            temp = self.conS(x)
            temp = self.relu(temp)
            temp = self.conT(temp)
        elif self.conv_mode == 'b':
            temp_x = self.conS(x)
            temp_x = self.relu(temp_x)
            temp = self.conT(x)
            temp = temp + temp_x
        elif self.conv_mode == 'c':
            temp_x = self.conS(x)
            temp_x = self.relu(temp_x)
            temp = self.conT(temp_x)
            temp = temp + temp_x
        return temp

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size = 4, conv_mode = 'a', batch_norm = False, drop_rate = 0.5):
        super(_DenseLayer,self).__init__()
        self.conv_mode = conv_mode.lower()
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate
        self.relu = nn.ReLU(inplace=True)
        if conv_mode == '2d':
            self.conv1 = nn.Conv2d(in_channels, bn_size*growth_rate, 1, 1, bias=False)
            if batch_norm:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.bn2 = nn.BatchNorm2d(growth_rate*bn_size)
            self.dropout = nn.Dropout2d(drop_rate)
        else:
            self.conv1 = nn.Conv3d(in_channels, bn_size * growth_rate, 1, 1, bias=False)
            if batch_norm:
                self.bn1 = nn.BatchNorm3d(in_channels)
                self.bn2 = nn.BatchNorm3d(growth_rate*bn_size)
            self.dropout = nn.Dropout3d(drop_rate)
        self.conv2 = _P3Dconv(growth_rate * bn_size, growth_rate, conv_mode = conv_mode)

    def forward(self, x):
        if self.batch_norm:
            temp = self.bn1(x)
        else:
            temp = x
        temp = self.relu(temp)
        temp = self.conv1(temp)
        if self.batch_norm:
            temp = self.bn2(temp)
        temp = self.relu(temp)
        temp = self.conv2(temp)
        if self.drop_rate > 0:
            temp = self.dropout(temp)
        return torch.cat([x,temp],1)

class _DenseBlock(nn.Module):
    def __init__(self,in_channels,growth_rate,num_of_lauer, tf_conv3d, batch_norm = False, bn_size = 4, drop_rate = 0.5, ST_struc=['a','b','c']):
        super(_DenseBlock,self).__init__()
        if not tf_conv3d:
            ST_struc = ['2d']
        len_ST = len(ST_struc)
        curr_channels = in_channels
        self.features = nn.Sequential()
        for id in range(num_of_lauer):
            self.features.add_module(f'DenseLayer{id}', _DenseLayer(curr_channels, growth_rate, bn_size = bn_size, conv_mode=ST_struc[id % len_ST],batch_norm=batch_norm,drop_rate=drop_rate))
            curr_channels = curr_channels + growth_rate
    def forward(self,x):
        return self.features(x)


        


class _Transition(nn.Module):
    def __init__(self,in_channels,out_channels,batch_norm = False, tf_conv3d = True):
        super(_Transition,self).__init__()
        self.tf_conv3d = tf_conv3d
        self.batch_norm = batch_norm
        if tf_conv3d:
            if self.batch_norm:
                self.bn = nn.BatchNorm3d(in_channels)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.pool = nn.AvgPool3d(kernel_size=(2,2,2))
        else:
            if self.batch_norm:
                self.bn = nn.BatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.pool = nn.AvgPool2d(kernel_size = (2,2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,in_channels,in_frames,init_channels,growth_rate,blocks,num_classes=2,drop_rate=0.5, bn_size = 4, batch_norm = False):
        super(DenseNet,self).__init__()
        self.batch_norm = batch_norm
        self.in_frames = in_frames
        self.preconv = nn.Sequential()
        self.preconv.add_module('conv0', nn.Conv3d(in_channels, init_channels, kernel_size = (1,7,7), stride = (1,2,2), padding = (0,3,3), bias = False))
        if batch_norm:
            self.preconv.add_module('batchnorm0',nn.BatchNorm3d(init_channels))
        self.preconv.add_module('relu0', nn.ReLU(inplace=True))
        self.preconv.add_module('maxpool0', nn.MaxPool3d(kernel_size = (1,3,3), stride = (1,2,2), padding=(0,1,1)))
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size = (1,3,3), stride = (1,2,2), padding = (0,1,1)), #250
            nn.ReLU(inplace=True),
            nn.Conv3d(init_channels, init_channels, kernel_size = (1,3,3), stride = (1,2,2), padding = (0,1,1)), #125
            nn.ReLU(inplace=True),
            nn.Conv3d(init_channels, init_channels, kernel_size = (1,3,3), stride = (1,2,2), padding = (0,1,1)), #62
            nn.ReLU(inplace=True),
            nn.Conv3d(init_channels, init_channels, kernel_size = (1,3,3), stride = (1,2,2), padding = (0,1,1)), #31
            nn.ReLU(inplace=True),
            nn.Conv3d(init_channels, init_channels, kernel_size = (1,3,3), stride = (1,2,2), padding = (0,1,1)), #15
            nn.ReLU(inplace=True),
        )
        curr_depth = in_frames
        curr_features = init_channels
        self.Conv_3d = nn.Sequential()
        self.Conv_2d = nn.Sequential()
        for i, num in enumerate(blocks):
            if curr_depth > 1:
                self.Conv_3d.add_module(f'DenseBlock{i+1}', _DenseBlock(curr_features, growth_rate, num, tf_conv3d = True, batch_norm=batch_norm, drop_rate=drop_rate))
                curr_features = curr_features + num * growth_rate
                self.Conv_3d.add_module(f'Transition{i+1}', _Transition(curr_features, curr_features//2, batch_norm=batch_norm, tf_conv3d=True))
                curr_features = curr_features//2
                curr_depth = curr_depth // 2
            else:
                self.Conv_2d.add_module(f'DenseBlock{i+1}', _DenseBlock(curr_features, growth_rate, num, tf_conv3d = False, batch_norm=batch_norm, drop_rate=drop_rate))
                curr_features = curr_features + num * growth_rate
                if i < len(blocks)-1:
                    self.Conv_2d.add_module(f'Transition{i+1}', _Transition(curr_features, curr_features//2, batch_norm=batch_norm, tf_conv3d=False))
                    curr_features = curr_features // 2
        self.avgpool_final = nn.AvgPool2d(kernel_size=(15,15))
        self.avgpool_finalr = nn.AvgPool2d(kernel_size=(16,16))
        self.fc = nn.Linear(curr_features, num_classes)
        self.fcr = nn.Sequential(
            nn.Linear(init_channels, 3),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()
        # r = 230
        # y,x = np.ogrid[0:500, 0:500]
        # mask0 = ((x-250)**2 + (y-250)**2 <= r**2) * 1.0
        # self.mask = torch.from_numpy(mask0.astype('float32')).cuda()
        
    def forward(self, x):

        attn_map_r_xy = self.attention(x)
        attn_map_r_xy  = self.avgpool_finalr(torch.squeeze(attn_map_r_xy))
        sizes = attn_map_r_xy.size()
        attn_map_r_xy = attn_map_r_xy.view(sizes[0], -1)
        attn_map_r_xy = self.fcr(attn_map_r_xy)

        attn_map_r = attn_map_r_xy[:,0] * 70 + 160
        attn_map_shift_x = attn_map_r_xy[:,1] * 20 - 10
        attn_map_shift_y = attn_map_r_xy[:,2] * 20 - 10

        device = torch.device('cuda:0')
        # attn_map_r = torch.maximum(attn_map_r, torch.ones([sizes[0], 1]).to(device)*100)

        attn_map = torch.zeros([sizes[0],1,1,500,500], dtype = torch.float32).to(device)

        x0 = torch.linspace(-250, 250, steps=500).to(device)
        y0 = torch.linspace(-250, 250, steps=500).to(device)
        x1,y1 = torch.meshgrid(x0, y0)
        
        for kk in range(sizes[0]):     
            cur_r = (attn_map_r[kk])
            cur_shift_x = attn_map_shift_x[kk]
            cur_shift_y = attn_map_shift_y[kk]
            mask0 = self.sigmoid(((cur_r)**2 - ((x1-cur_shift_x)**2 + (y1-cur_shift_y)**2))/10000)
            attn_map[kk,0,0,:,:] = mask0
        # attn_map = (attn_map - torch.amin(attn_map, dim=(1, 2, 3, 4), keepdim=True))/(torch.amax(attn_map, dim=(1, 2, 3, 4), keepdim=True) - torch.amin(attn_map, dim=(1, 2, 3, 4), keepdim=True))
        # attn_map = (attn_map>torch.mean(attn_map)).float() * self.mask
        # attn_map = (binarize(attn_map) - attn_map).detach() + attn_map
        x = x * attn_map
        # x = (x * attn_map - x).detach() + x
        x = self.preconv(x)
        # attn_map = self.attention(x)
        # attn_map_tosave = F.interpolate(attn_map, size=(1,500,500))
        # x = x * attn_map
        x = self.Conv_3d(x)
        sizes=x.size()
        x = x.view(-1,sizes[1],sizes[3],sizes[4])
        x = self.Conv_2d(x)
        x = self.avgpool_final(x)
        sizes = x.size()
        x = x.view(sizes[0], -1)
        x = self.fc(x)

        return x, attn_map, attn_map_r, attn_map_shift_x, attn_map_shift_y 
    
if __name__ == '__main__':
    data = torch.autograd.Variable(torch.rand(1, 10, 500, 500))  # torch.rand(10, 2, 8, 288, 288)
    model = DenseNet(in_channels = 1, in_frames = 10, init_channels = 64,growth_rate = 32,blocks = [3,4,6,3],num_classes=2,drop_rate=0.5, bn_size = 4, batch_norm = False)

