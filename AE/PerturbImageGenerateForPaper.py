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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import foolbox as fb
import ResNet

class SimpleDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image)
        sample = {"Images": image}
        return sample

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet.ResNet().to(device)
    models_dir = './models'
    file_name = '/ResNet_CIFAR10.pth'
    model.load_state_dict(torch.load(models_dir + file_name))
    model.eval()

    bounds = (-0.4914/0.1994, 1.4914/0.1994)
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    
    success = 0
    outputs = []
    for data in trainloader:
        output = []
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        output.append(inputs[0].cpu().detach().numpy())

        attack = fb.attacks.FGSM()
        raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
        if is_adv:
            output.append(clipped[0].cpu().detach().numpy())
        
        attack = fb.attacks.PGD()
        raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
        if is_adv:
            output.append(clipped[0].cpu().detach().numpy())
        
        attack = fb.attacks.LinfDeepFoolAttack()
        raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
        if is_adv:
            output.append(clipped[0].cpu().detach().numpy())
        
        if len(output) == 4:
            success += 1
            for i in output:
                outputs.append(i)
            if success == 8:
                break
    
    # print(output)
    t_images_labels_df = pd.DataFrame({'Images': outputs})
    t_dataset = SimpleDataset(t_images_labels_df['Images'])
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=False)

    for data in t_dataloader:
        image = data["Images"]
        print(image.shape)
        plt.imshow(np.transpose(torchvision.utils.make_grid(image).cpu().numpy(), (1, 2, 0)))
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show()

        