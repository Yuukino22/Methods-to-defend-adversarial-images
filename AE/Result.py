import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PerturbDataGenerate import AttackDataset
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder
from ResNet import ResNet
from TrainDataGenerate import MyDataset


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100
    models_dir = './models'

    AE = AutoEncoder().to(device)
    RN = ResNet().to(device)

    AE.load_state_dict(torch.load(models_dir + '/AutoEncoder_CIFAR10.pth'))
    AE.eval()

    RN.load_state_dict(torch.load(models_dir + '/ResNet_CIFAR10.pth'))
    RN.eval()


    test = torch.load(models_dir + '/Test_10000_3000_3000_4000.pt')
    #test = torch.load(models_dir + '/Test_0_5000_0_0.pt')
    #test = torch.load(models_dir + '/Test_0_0_5000_0.pt')
    #test = torch.load(models_dir + '/Test_0_0_0_5000.pt')
    test_ds = MyDataset(test['Images'],test['Labels'])
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)



    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            #hidden, out =  AE(images)
            outputs = RN(images)
            _, result = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (result == labels).sum().item()
        print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))
    
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, out =  AE(images)
            outputs = RN(out)
            _, result = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (result == labels).sum().item()
        print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))

    
    

    


