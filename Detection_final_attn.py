# from __future__ import print_function
from configparser import Interpolation
from torch.utils import data
import random
import os
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import scipy.ndimage as ndimage
import getopt
import sys
from configobj import ConfigObj
from tqdm import tqdm
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torchvision.utils as vutils
import torch.nn.functional as F
import pickle
import network_v3_attentionr_soft_xy as network
# import network_v3 as network
from matplotlib import pyplot as plt
# import cv2

import shutil

def is_target_file(filename):
    return filename.endswith("npy")


def load_img(filepath):
    y = np.load(filepath).astype(np.float32)
    return y

class DataFolder(data.Dataset):


    def __init__(self,in_channels,in_frames,image_dir,lab,start_frame = 0, input_transform=None):
        super(DataFolder,self).__init__()
        self.image_filenames = []
        self.label = []
        self.in_channels = in_channels
        self.in_frames = in_frames
        self.start_frame = start_frame
        cur_label = np.asarray(lab)
        for inner_dir in image_dir:
            for x in listdir(inner_dir):
                if is_target_file(x):
                    self.image_filenames.append(join(inner_dir,x))
                    self.label.append(cur_label)
        self.input_transform = input_transform
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,index):
        path = self.image_filenames[index]

        if self.start_frame == -1:
            s = random.randint(0,5)
        else:
            s = self.start_frame
        input0 = load_img(path)[:,:,s:self.in_frames+s].astype(float)

        label = self.label[index]

        r = 230
        input = np.zeros((1, self.in_frames, 500, 500))
        mask = np.zeros((500, 500, self.in_frames))
        y,x = np.ogrid[0:np.shape(input0)[0], 0:np.shape(input0)[1]]
        mask0 = ((x-np.shape(input0)[0]//2)**2 + (y-np.shape(input0)[1]//2)**2 <= r**2) * 1.0
        for kk in range(self.in_frames):
            mask[:,:,kk] = mask0
        input_center = input0 * mask

        input[0,:,:,:] = np.transpose(input_center, (2,0,1)) # 10*550*550 -> 1*10*550*500
        img = input.copy()

        return img,label,path

def get_data_set(in_channels,in_frames,image_dir,lab,start_frame = 0, input_transform=None):
    return DataFolder(in_channels,in_frames,image_dir,lab,start_frame,input_transform)


if __name__ == '__main__':

    modelPath = r'Y:\Swarming_copy_network_training\Network_Code\r230_attn_20240213_10frame_boundr_soft_xy_shift10_avefirst_init8_growth4_3363_0.5_batch32\model_detection_epoch44.pth'

    test_path = r'Y:\Swarming_copy_network_training\Network_Code\GitHub\TestImgs' 

    result_file = os.path.join(test_path,"results_attn_20240218_init8_growth4_3363_0.5_batch32_v1_epoch44.txt")
    result_file = open(result_file,'w')
        
    input_png_file = test_path.replace('npy','Tiff')

    batch_size = 8
    model = network.DenseNet(in_channels = 1, in_frames = 1, init_channels = 8, growth_rate = 4,blocks = [3,3,6,3],num_classes=2, drop_rate=.5, bn_size = 8 , batch_norm = True) #bn_size was 4 final model
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    model.to(device)
    model.load_state_dict(torch.load(modelPath, map_location='cuda:0'))
    model.eval()
    
    image_save_path = test_path
    dataset = get_data_set(in_channels = 1, in_frames = 20, image_dir = [test_path], lab = 0, start_frame = 0)
    # 
    data_loader = DataLoader(dataset=dataset, num_workers=6,
                        batch_size=batch_size, shuffle=False, drop_last=False,
                        pin_memory=True)

    input = torch.Tensor(batch_size, 1, 1, 500, 500).to(device)

    score_total = 0
    count = 0
    with torch.no_grad():
        for i,batch in enumerate(tqdm(data_loader)):

            starttime = time.time()

            inputallframe = batch[0].float().to(device)

            inputallframe_5frame = inputallframe[:, :, :5,:, :]
            input_5frame = torch.mean(inputallframe_5frame, 2, True)

            inputallframe = inputallframe[:, :, :10,:, :]
            input = torch.mean(inputallframe, 2, True)

            # added at 02/13/2024
            r = 230
            y,x = np.ogrid[0:np.shape(input)[-1], 0:np.shape(input)[-2]]
            mask0 = ((x-np.shape(input)[-1]//2)**2 + (y-np.shape(input)[-2]//2)**2 <= r**2) * 1.0
            mask0 = torch.from_numpy(np.expand_dims(mask0, axis=(0,1))).to(device)

            for iii in range(input.size(dim=0)):
                input_centerlist = torch.reshape(input[iii,:,:,:,:], (1,-1))
                input_centerlist = input_centerlist[input_centerlist.nonzero(as_tuple=True)]

                # calculate mean and std of center area
                input_mean = torch.mean(input_centerlist)
                input_std = torch.std(input_centerlist)

                input_norm = (input[iii,:,:,:,:] - input_mean)/(input_std+1e-8)
                input[iii,:,:,:,:] = input_norm * mask0

            # input = inputallframe
                
            path = batch[2]
            output, feature_map, attn_map_r, attn_map_shift_x, attn_map_shift_y  = model(input)
            # output, feature_map  = model(input)
            _,predicted = torch.max(output,1)

            endtime = time.time()
            print(f'Inference time: {endtime-starttime}')

            for j,lab in enumerate(predicted):
                filename = os.path.basename(path[j])
                name,ext = os.path.splitext(filename)
                score = torch.exp(output[j,:]) / (torch.exp(output[j,0])+torch.exp(output[j,1]))
                result_file.write(f'{filename} {score[0]} {score[1]}\n')
                if score[0] > 0.5:
                    result_file.write("test_results: non-swarming(planktonic)\n")
                else:
                    result_file.write("test_results: swarming\n")