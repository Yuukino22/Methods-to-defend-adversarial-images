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
from generate_latent_space import LatentDataset
from torch.utils.data import Dataset, DataLoader

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 48*4*4)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SimpleNet Train and Valid')
    parser.add_argument("--v", action="store_true", default=False, help="Perform validation only.")
    parser.add_argument("--t", action="store_true", default=False, help="Perform test only.")
    parser.add_argument("--train", default = "Train_NF_50000_15000_15000_20000.pt")
    parser.add_argument("--valid", default = "Valid_NF_50000_15000_15000_20000.pt")
    parser.add_argument("--test", default = "Test_NF_10000_3000_3000_4000.pt")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_dir = './output'
    batch_size = 64

    test_pt = torch.load(models_dir + '/{}'.format(args.test))
    test_ds = LatentDataset(test_pt['Images'],test_pt['Labels'])
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    valid_pt = torch.load(models_dir + '/{}'.format(args.valid))
    valid_ds = LatentDataset(valid_pt['Images'],valid_pt['Labels'])
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    train_pt = torch.load(models_dir + '/{}'.format(args.train))
    train_ds = LatentDataset(train_pt['Images'],train_pt['Labels'])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    num_epochs = 25
    model = SimpleNet().to(device)
    models_dir = './output'
    file_name = '/SimpleNet_{}.pth.'.format(num_epochs)

    if args.t:
        model.load_state_dict(torch.load(models_dir + file_name))
        model.eval()
        print("Start Testing!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_dl:
                images = data["Images"]
                labels = data["Labels"]
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, result = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (result == labels).sum().item()
        print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))

        exit(0)

    if args.v:
        model.load_state_dict(torch.load(models_dir + file_name))
        model.eval()
        print("Start Validation!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in valid_dl:
                images = data["Images"]
                labels = data["Labels"]
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, result = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (result == labels).sum().item()
        print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))

        exit(0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dl):
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                print('Epoch {} Iteration {}, Loss: {}'.format(epoch, i, loss.data.cpu().numpy()))
    
    model.eval()
    print("Start Testing!")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in valid_dl:
            images = data["Images"]
            labels = data["Labels"]
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, result = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (result == labels).sum().item()
    print('Total: {}, Correct: {}, Percentage: {}%'.format(total, correct, correct * 100 / total))

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(model.state_dict(), models_dir + file_name)