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

class AttackDataset(Dataset):
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

    parser = argparse.ArgumentParser(description='Generate_perturbed_image')
    parser.add_argument("--test", action="store_true", default=False, help="Generate test data")
    parser.add_argument("--clean", action="store_true", default=False, help="Clean Image")
    parser.add_argument("--fgsm", action="store_true", default=False, help="FGSM Attack")
    parser.add_argument("--pgd", action="store_true", default=False, help="PGD Attack")
    parser.add_argument("--deepfool", action="store_true", default=False, help="Deep Fool Attack")
    args = parser.parse_args()


    transform = transforms.Compose([transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet.ResNet().to(device)
    models_dir = './models'
    file_name = '/ResNet_CIFAR10.pth'
    model.load_state_dict(torch.load(models_dir + file_name))
    model.eval()

    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    df_name = []
    df_total = []

    if args.clean:
        CLEANimages = []
        CLEANlabels = []
        total = 0
        if not args.test:
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                total += labels.size(0)
                for i in range(labels.size(0)):
                    CLEANimages.append(inputs[i].detach().cpu().numpy())
                    CLEANlabels.append(labels[i].detach().cpu().numpy())
            print("Train Clean data Complete.")
        else:
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                total += labels.size(0)
                for i in range(labels.size(0)):
                    CLEANimages.append(inputs[i].detach().cpu().numpy())
                    CLEANlabels.append(labels[i].detach().cpu().numpy())
            print("Test Clean data Complete.")
        
        clean_images_labels_df = pd.DataFrame({'Images': CLEANimages, 'Labels': CLEANlabels})
        clean_dataset = AttackDataset(clean_images_labels_df['Images'],clean_images_labels_df['Labels'])
        clean_dataloader = DataLoader(clean_dataset, batch_size=2, shuffle=True)


        if not args.test:
            df_name.append('/Clean.pt')
        else:
            df_name.append('/Cleantest.pt')
        df_total.append(clean_images_labels_df)

    if args.fgsm:
        FGSMimages = []
        FGSMlabels = []
        attack = fb.attacks.FGSM()
        total = 0
        if not args.test:
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
                
                total += labels.size(0)
                success = 0
                for i in range(labels.size(0)):
                    if is_adv[i]:
                        FGSMimages.append(clipped[i].detach().cpu().numpy())
                        FGSMlabels.append(labels[i].detach().cpu().numpy())
                        success += 1
                print('FGSM: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))

            success = len(FGSMlabels)
            print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))
        else:
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
                
                total += labels.size(0)
                success = 0
                for i in range(labels.size(0)):
                    if is_adv[i]:
                        FGSMimages.append(clipped[i].detach().cpu().numpy())
                        FGSMlabels.append(labels[i].detach().cpu().numpy())
                        success += 1
                print('FGSM: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))

            success = len(FGSMlabels)
            print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))



        fgsm_images_labels_df = pd.DataFrame({'Images': FGSMimages, 'Labels': FGSMlabels})
        fgsm_dataset = AttackDataset(fgsm_images_labels_df['Images'],fgsm_images_labels_df['Labels'])
        fgsm_dataloader = DataLoader(fgsm_dataset, batch_size=2, shuffle=True)


        if not args.test:
            df_name.append('/FGSM.pt')
        else:
            df_name.append('/FGSMtest.pt')
        df_total.append(fgsm_images_labels_df)
    
    if args.pgd:
        PGDimages = []
        PGDlabels = []
        attack = fb.attacks.PGD()
        total = 0
        if not args.test:
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
                
                total += labels.size(0)
                success = 0
                for i in range(labels.size(0)):
                    if is_adv[i]:
                        PGDimages.append(clipped[i].detach().cpu().numpy())
                        PGDlabels.append(labels[i].detach().cpu().numpy())
                        success += 1
                print('PGD: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))


            success = len(PGDlabels)
            print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))
        
        else:
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons=0.02)
                
                total += labels.size(0)
                success = 0
                for i in range(labels.size(0)):
                    if is_adv[i]:
                        PGDimages.append(clipped[i].detach().cpu().numpy())
                        PGDlabels.append(labels[i].detach().cpu().numpy())
                        success += 1
                print('PGD: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))


            success = len(PGDlabels)
            print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))


        pgd_images_labels_df = pd.DataFrame({'Images': PGDimages, 'Labels': PGDlabels})
        pgd_dataset = AttackDataset(pgd_images_labels_df['Images'],pgd_images_labels_df['Labels'])
        pgd_dataloader = DataLoader(pgd_dataset, batch_size=2, shuffle=True)

        if not args.test:
            df_name.append('/PGD.pt')
        else:
            df_name.append('/PGDtest.pt')
        df_total.append(pgd_images_labels_df)     
    
    if args.deepfool:
        DFimages = []
        DFlabels = []
        attack = fb.attacks.LinfDeepFoolAttack()
        total = 0
        if not args.test:
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons = 0.02)
                
                total += labels.size(0)
                success = 0
                for i in range(labels.size(0)):
                    if is_adv[i]:
                        DFimages.append(clipped[i].detach().cpu().numpy())
                        DFlabels.append(labels[i].detach().cpu().numpy())
                        success += 1
                print('DeepFool: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))


            success = len(DFlabels)
            print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))
        else:
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                raw, clipped, is_adv = attack(fmodel, inputs, labels, epsilons = 0.01)
                
                total += labels.size(0)
                success = 0
                for i in range(labels.size(0)):
                    if is_adv[i]:
                        DFimages.append(clipped[i].detach().cpu().numpy())
                        DFlabels.append(labels[i].detach().cpu().numpy())
                        success += 1
                print('DeepFool: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / labels.size(0)))


            success = len(DFlabels)
            print('Summary: Success attack {}, Total {}, Percentage {}'.format(success, total, success * 100 / total))


        df_images_labels_df = pd.DataFrame({'Images': DFimages, 'Labels': DFlabels})
        df_dataset = AttackDataset(df_images_labels_df['Images'],df_images_labels_df['Labels'])
        df_dataloader = DataLoader(df_dataset, batch_size=2, shuffle=True)

        if not args.test:
            df_name.append('/DeepFool.pt')
        else:
            df_name.append('/DeepFooltest.pt')
        df_total.append(df_images_labels_df)



    for i in range(len(df_name)):
        torch.save(df_total[i], models_dir + df_name[i])

    # FF = torch.load(models_dir + '/DeepFool.pt')
    # TD = AttackDataset(FF['Images'],FF['Labels'])
    # DL_DS = DataLoader(TD, batch_size=1, shuffle=True)

    # for data in DL_DS:
    #     images = data["Images"]
    #     labels = data["Labels"]
    #     print(images.shape)
    #     plt.imshow(np.transpose(torchvision.utils.make_grid(images).cpu().numpy(), (1, 2, 0)))
    #     plt.show()
    #     break


    # t = 0
    # for i in DL_DS:
    #     t += 1
    #     if t == 30000:
    #         break

    # s = 0
    # for i in DL_DS:
    #     s += 1
    # print(t, s)
