"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import sys
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
sys.path.append('..') 
from utils.pointconv_util import PointConvDensitySetAbstraction
from utils.utils import *

class PointConvDensityClsSsg(nn.Module):
    def __init__(self, num_classes = 40, pruning_rate=[0.]*10):
        super(PointConvDensityClsSsg, self).__init__()
        feature_dim = 3

        mlp1 = [64, 64, 128]
        mlp2 = [128, 128, 256]
        mlp3 = [256, 512, 1024]

        self.mlp1 = [int(x*(1.-y)) for x,y in zip(mlp1,pruning_rate[0:3])]
        self.mlp2 = [int(x*(1.-y)) for x,y in zip(mlp2,pruning_rate[3:6])]
        self.mlp3 = [int(x*(1.-y)) for x,y in zip(mlp3,pruning_rate[6:9])]
        

        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=self.mlp1, bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=self.mlp1[-1] + 3, mlp=self.mlp2, bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=self.mlp2[-1] + 3, mlp=self.mlp3, bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(self.mlp3[-1], 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, self.mlp3[-1])
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

def all_layers(model):
    layers = []
    _name_ = ['weight','bias','running_mean','running_var','num_batches_tracked']
    state_dict = model.state_dict()
    for k,name in enumerate(state_dict):
        # print(k,name,state_dict[name].size())
        if any(x in name for x in _name_):
            layers.append(name)
    # print(len(layers),layers)
    return layers

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    pruning_rate = [0.2]*10
    model = PointConvDensityClsSsg(num_classes=40,pruning_rate=pruning_rate)

    print(model)
    
    input = torch.randn((8,6,1024))
    label = torch.randn(8,16)
    output= model(input[:,:3,:],input[:,3:,:])
    print(output.size())
    
    
    state_dict = model.state_dict()
    for k,name in enumerate(state_dict):
        print(k,name,state_dict[name].size()) #,type(state_dict[name]

    model_info(model,npoints=1024)





