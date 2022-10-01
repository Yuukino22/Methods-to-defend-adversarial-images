import json
import torch
from torchvision.utils import make_grid
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

from datasets import get_CIFAR10, preprocess, postprocess
from model import Glow

class LatentDataset(Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        image = torch.tensor(image)
        label = torch.tensor(label)
        sample = {"Images": image, "Labels": label}
        return sample

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate_Latent_Space')
    parser.add_argument("--test", action="store_true", default=False, help="Generate test data")
    parser.add_argument("--clean", action="store_true", default=False, help="Clean Image")
    parser.add_argument("--fgsm", action="store_true", default=False, help="FGSM Attack")
    parser.add_argument("--pgd", action="store_true", default=False, help="PGD Attack")
    parser.add_argument("--deepfool", action="store_true", default=False, help="Deep Fool Attack")
    args = parser.parse_args()

    device = torch.device("cuda")

    output_folder = 'output/'
    model_name = 'glow_affine_coupling.pt'

    batch_size = 64

    with open(output_folder + 'hparams.json') as json_file:  
        hparams = json.load(json_file)
        
    image_shape = (32, 32, 3)
    num_classes = 10

    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                hparams['learn_top'], hparams['y_condition'])

    model.load_state_dict(torch.load(output_folder + model_name))
    model.set_actnorm_init()

    model = model.to(device)
    model = model.eval()

    transform = transforms.Compose([transforms.ToTensor(), preprocess])

    trainset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2)

    df_name = []
    df_total = []

    if args.clean:

        Clean_Latent = []
        Clean_Label = []
        
        if not args.test:
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                hidden, logdet, _ = model(inputs)
                for j in range(len(hidden)):
                    Clean_Latent.append(hidden[j].cpu().detach().numpy())
                    Clean_Label.append(0)
                print('Clean, Iteration {}'.format(i))
        else:
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                hidden, logdet, _ = model(inputs)
                for j in range(len(hidden)):
                    Clean_Latent.append(hidden[j].cpu().detach().numpy())
                    Clean_Label.append(0)
                print('Clean, Iteration {}'.format(i))

        Clean_images_labels_df = pd.DataFrame({'Images': Clean_Latent, 'Labels': Clean_Label})
        Clean_dataset = LatentDataset(Clean_images_labels_df['Images'],Clean_images_labels_df['Labels'])
        Clean_dataloader = DataLoader(Clean_dataset, batch_size=2, shuffle=True)
    
        if not args.test:
            df_name.append('/Clean_Latent.pt')
        else:
            df_name.append('/Clean_Latent_test.pt')
        df_total.append(Clean_images_labels_df)
    
    if args.fgsm:
        if not args.test:
            fgsm = torch.load(output_folder + '/FGSM.pt')
        else:
            fgsm = torch.load(output_folder + '/FGSMtest.pt')

        fgsm_ds = AttackDataset(fgsm['Images'],fgsm['Labels'])
        fgsm_dl = DataLoader(fgsm_ds, batch_size=batch_size, shuffle=True)

        FGSM_Latent = []
        FGSM_Label = []
        
        for i, data in enumerate(fgsm_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, logdet, _ = model(images)
            for j in range(len(hidden)):
                FGSM_Latent.append(hidden[j].cpu().detach().numpy())
                FGSM_Label.append(1)
            print('FGSM, Iteration {}'.format(i))

        fgsm_images_labels_df = pd.DataFrame({'Images': FGSM_Latent, 'Labels': FGSM_Label})
        fgsm_dataset = LatentDataset(fgsm_images_labels_df['Images'],fgsm_images_labels_df['Labels'])
        fgsm_dataloader = DataLoader(fgsm_dataset, batch_size=2, shuffle=True)
    
        if not args.test:
            df_name.append('/FGSM_Latent.pt')
        else:
            df_name.append('/FGSM_Latent_test.pt')
        df_total.append(fgsm_images_labels_df)
    
    if args.pgd:
        if not args.test:
            pgd = torch.load(output_folder + '/PGD.pt')
        else:
            pgd = torch.load(output_folder + '/PGDtest.pt')

        pgd_ds = AttackDataset(pgd['Images'],pgd['Labels'])
        pgd_dl = DataLoader(pgd_ds, batch_size=batch_size, shuffle=True)

        PGD_Latent = []
        PGD_Label = []
        
        for i, data in enumerate(pgd_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, logdet, _ = model(images)
            for j in range(len(hidden)):
                PGD_Latent.append(hidden[j].cpu().detach().numpy())
                PGD_Label.append(1)
            print('PGD, Iteration {}'.format(i))

        pgd_images_labels_df = pd.DataFrame({'Images': PGD_Latent, 'Labels': PGD_Label})
        pgd_dataset = LatentDataset(pgd_images_labels_df['Images'],pgd_images_labels_df['Labels'])
        pgd_dataloader = DataLoader(pgd_dataset, batch_size=2, shuffle=True)
        
        if not args.test:
            df_name.append('/PGD_Latent.pt')
        else:
            df_name.append('/PGD_Latent_test.pt')
        df_total.append(pgd_images_labels_df)  
    
    if args.deepfool:
        if not args.test:
            deepfool = torch.load(output_folder + '/DeepFool.pt')
        else:
            deepfool = torch.load(output_folder + '/DeepFooltest.pt')

        deepfool_ds = AttackDataset(deepfool['Images'],deepfool['Labels'])
        deepfool_dl = DataLoader(deepfool_ds, batch_size=batch_size, shuffle=True)

        deepfool_Latent = []
        deepfool_Label = []
        
        for i, data in enumerate(deepfool_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            hidden, logdet, _ = model(images)
            for j in range(len(hidden)):
                deepfool_Latent.append(hidden[j].cpu().detach().numpy())
                deepfool_Label.append(1)
            print('DeepFool, Iteration {}'.format(i))

        deepfool_images_labels_df = pd.DataFrame({'Images': deepfool_Latent, 'Labels': deepfool_Label})
        deepfool_dataset = LatentDataset(deepfool_images_labels_df['Images'],deepfool_images_labels_df['Labels'])
        deepfool_dataloader = DataLoader(deepfool_dataset, batch_size=2, shuffle=True)

        if not args.test:
            df_name.append('/DeepFool_Latent.pt')
        else:
            df_name.append('/DeepFool_Latent_test.pt')
        df_total.append(deepfool_images_labels_df)  

    for i in range(len(df_name)):
        torch.save(df_total[i], output_folder + df_name[i])







