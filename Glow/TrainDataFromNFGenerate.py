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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Data From AutoEncoder Generate')
    parser.add_argument("--CLEAN", type=int, default=50000, help="Clean Image num")
    parser.add_argument("--FGSM", type=int, default=15000, help="FGSM Image num")
    parser.add_argument("--PGD", type=int, default=15000, help="PGD Image num")
    parser.add_argument("--DF", type=int, default=20000, help="DeepFool Image num")
    parser.add_argument("--train", action="store_true", default=False, help="Train only.")
    parser.add_argument("--test", action="store_true", default=False, help="Test only.")
    args = parser.parse_args()

    split = [args.CLEAN, args.FGSM, args.PGD, args.DF]
    split_test = [int(args.CLEAN / 5), int(args.FGSM / 5), int(args.PGD / 5), int(args.DF / 5)]

    models_dir = './output'
    batch_size = 50

    clean = torch.load(models_dir + '/Clean_Latent.pt')
    clean_ds = LatentDataset(clean['Images'],clean['Labels'])
    clean_dl = DataLoader(clean_ds, batch_size=batch_size, shuffle=True)

    fgsm = torch.load(models_dir + '/FGSM_Latent.pt')
    fgsm_ds = LatentDataset(fgsm['Images'],fgsm['Labels'])
    fgsm_dl = DataLoader(fgsm_ds, batch_size=batch_size, shuffle=True)

    pgd = torch.load(models_dir + '/PGD_Latent.pt')
    pgd_ds = LatentDataset(pgd['Images'],pgd['Labels'])
    pgd_dl = DataLoader(pgd_ds, batch_size=batch_size, shuffle=True)

    deepfool = torch.load(models_dir + '/DeepFool_Latent.pt')
    deepfool_ds = LatentDataset(deepfool['Images'],deepfool['Labels'])
    deepfool_dl = DataLoader(deepfool_ds, batch_size=batch_size, shuffle=True)

    clean_test = torch.load(models_dir + '/Clean_Latent_test.pt')
    clean_test_ds = LatentDataset(clean_test['Images'],clean_test['Labels'])
    clea_test_dl = DataLoader(clean_test_ds, batch_size=batch_size, shuffle=True)

    fgsm_test = torch.load(models_dir + '/FGSM_Latent_test.pt')
    fgsm_test_ds = LatentDataset(fgsm_test['Images'],fgsm_test['Labels'])
    fgsm_test_dl = DataLoader(fgsm_test_ds, batch_size=batch_size, shuffle=True)

    pgd_test = torch.load(models_dir + '/PGD_Latent_test.pt')
    pgd_test_ds = LatentDataset(pgd_test['Images'],pgd_test['Labels'])
    pgd_test_dl = DataLoader(pgd_test_ds, batch_size=batch_size, shuffle=True)

    deepfool_test = torch.load(models_dir + '/DeepFool_Latent_test.pt')
    deepfool_test_ds = LatentDataset(deepfool_test['Images'],deepfool_test['Labels'])
    deepfool_test_dl = DataLoader(deepfool_test_ds, batch_size=batch_size, shuffle=True)

    Train_Latent = []
    Train_Label = []

    Valid_Latent = []
    Valid_Label = []

    Test_Latent = []
    Test_Label = []

    if args.test:

        count_clean = 0
        for i, data in enumerate(clean_dl):
            if count_clean < split_test[0]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Test_Latent.append(images[j])
                    Test_Label.append(0)
                print('Clean Image, Test, Iteration {}'.format(i))
            else:
                break
            count_clean += batch_size
        
        count_fgsm = 0
        for i, data in enumerate(fgsm_test_dl):
            if count_fgsm < split_test[1]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Test_Latent.append(images[j])
                    Test_Label.append(1)
                print('FGSM Image, Test, Iteration {}'.format(i))
            else:
                break
            count_fgsm += batch_size
        
        count_pgd = 0
        for i, data in enumerate(pgd_test_dl):
            if count_pgd < split_test[2]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Test_Latent.append(images[j])
                    Test_Label.append(1)
                print('PGD Image, Test, Iteration {}'.format(i))
            else:
                break
            count_pgd += batch_size
        
        count_deepfool = 0
        for i, data in enumerate(deepfool_test_dl):
            if count_deepfool < split_test[3]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Test_Latent.append(images[j])
                    Test_Label.append(1)
                print('DeepFool Image, Test, Iteration {}'.format(i))
            else:
                break
            count_deepfool += batch_size
        
        Test_images_labels_df = pd.DataFrame({'Images': Test_Latent, 'Labels': Test_Label})
        Test_dataset = LatentDataset(Test_images_labels_df['Images'],Test_images_labels_df['Labels'])
        Test_dataloader = DataLoader(Test_dataset, batch_size=2, shuffle=True)

    if args.train:

        count_clean = 0
        for i, data in enumerate(clean_dl):
            if count_clean < split[0] * 0.9:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Train_Latent.append(images[j])
                    Train_Label.append(0)
                print('Clean Image, Train, Iteration {}'.format(i))
            elif count_clean >= split[0] * 0.9 and count_clean < split[0]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Valid_Latent.append(images[j])
                    Valid_Label.append(0)
                print('Clean Image, Valid, Iteration {}'.format(i))
            else:
                break
            count_clean += batch_size
        
        count_fgsm = 0
        for i, data in enumerate(fgsm_dl):
            if count_fgsm < split[1] * 0.9:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Train_Latent.append(images[j])
                    Train_Label.append(1)
                print('FGSM Image, Train, Iteration {}'.format(i))
            elif count_fgsm >= split[1] * 0.9 and count_fgsm < split[1]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Valid_Latent.append(images[j])
                    Valid_Label.append(1)
                print('FGSM Image, Valid, Iteration {}'.format(i))
            else:
                break
            count_fgsm += batch_size

        count_pgd = 0   
        for i, data in enumerate(pgd_dl):
            if count_pgd < split[2] * 0.9:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Train_Latent.append(images[j])
                    Train_Label.append(1)
                print('PGD Image, Train, Iteration {}'.format(i))
            elif count_pgd >= split[2] * 0.9 and count_pgd < split[2]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Valid_Latent.append(images[j])
                    Valid_Label.append(1)
                print('PGD Image, Valid, Iteration {}'.format(i))
            else:
                break
            count_pgd += batch_size

        count_deepfool = 0   
        for i, data in enumerate(deepfool_dl):
            if count_deepfool < split[3] * 0.9:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Train_Latent.append(images[j])
                    Train_Label.append(1)
                print('DeepFool Image, Train, Iteration {}'.format(i))
            elif count_deepfool >= split[3] * 0.9 and count_deepfool < split[3]:
                images = data["Images"]
                labels = data["Labels"]
                for j in range(batch_size):
                    Valid_Latent.append(images[j])
                    Valid_Label.append(1)
                print('DeepFool Image, Valid, Iteration {}'.format(i))
            else:
                break
            count_deepfool += batch_size
        
        Train_images_labels_df = pd.DataFrame({'Images': Train_Latent, 'Labels': Train_Label})
        Train_dataset = LatentDataset(Train_images_labels_df['Images'],Train_images_labels_df['Labels'])
        Train_dataloader = DataLoader(Train_dataset, batch_size=2, shuffle=True)

        Valid_images_labels_df = pd.DataFrame({'Images': Valid_Latent, 'Labels': Valid_Label})
        Valid_dataset = LatentDataset(Valid_images_labels_df['Images'],Valid_images_labels_df['Labels'])
        Valid_dataloader = DataLoader(Valid_dataset, batch_size=2, shuffle=True)

    
    if args.train:
        torch.save(Train_images_labels_df, models_dir + '/Train_NF_{}_{}_{}_{}.pt'.format(args.CLEAN, args.FGSM, args.PGD, args.DF))
        torch.save(Valid_images_labels_df, models_dir + '/Valid_NF_{}_{}_{}_{}.pt'.format(args.CLEAN, args.FGSM, args.PGD, args.DF))
    if args.test:
        torch.save(Test_images_labels_df, models_dir + '/Test_NF_{}_{}_{}_{}.pt'.format(split_test[0], split_test[1], split_test[2], split_test[3]))
    print("Successfully generate, Total {}".format(count_clean + count_fgsm + count_pgd + count_deepfool))