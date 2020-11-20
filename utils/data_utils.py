#!/usr/bin/env python3
# Please do not redistribute.

import torch
import numpy as np
import os
import urllib.request
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import tarfile

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def load_data(config):
    if 'CIFAR' in config['problem']:
        # Note: We still have to normalize and apply cutout later.
        if config['augment_data']:
            train_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])

        # Test data is not augmented.
        test_transform_stl_lsun = transforms.Compose([transforms.Resize((32,32)), 
                                                        transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
            
        # Magic dataset-specific normalization defined below.
        if config['problem'] == 'CIFAR10' \
           or config['problem'] == 'oodCIFAR10':
            dataset = dset.CIFAR10
            mean = (0.4914, 0.4822, 0.4465)
            std =  (0.2023, 0.1994, 0.2010)
            
        elif config['problem'] == 'CIFAR100' \
             or config['problem'] == 'oodCIFAR100':
            dataset = dset.CIFAR100
            mean =  (0.5071, 0.4867, 0.4408)
            std =  (0.2675, 0.2565, 0.2761)
        
        train_transform.transforms.append(transforms.Normalize(mean, std))
        test_transform.transforms.append(transforms.Normalize(mean, std))
        test_transform_stl_lsun.transforms.append(transforms.Normalize(mean,std))    

        if config['cutout_augment']:
            train_transform.transforms.append(Cutout(n_holes=1, length=16))

        if config['random_erasing']:
            train_transform.transforms.append(transforms.RandomErasing())

        train_data = dataset(config['data_folder'], train=True,
                                transform=train_transform, download=True)

        test_data = dataset(config['data_folder'], train=False,
                            transform=test_transform, download=True)

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config['test_batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True)

        train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=config['batch_size'], shuffle=True,
                num_workers=config['num_workers'], pin_memory=True)


        if config['problem'] == 'split-CIFAR':
            raise Exception('Currently unsupported.')

        elif config['problem'] == 'CIFAR10':
            assert(config['num_classes_per_task'] == 10)

        elif config['problem'] == 'CIFAR100':
            assert(config['num_classes_per_task'] == 100)

        elif 'oodCIFAR' in config['problem']:
            if config['problem'] == 'oodCIFAR100':
                assert(config['num_classes_per_task'] == 100)
                c_ood_test_data = dset.CIFAR10(config['data_folder'],
                                               train=False,
                                               transform=test_transform,
                                               download=True)
            else:
                assert(config['num_classes_per_task'] == 10)
                c_ood_test_data = dset.CIFAR100(config['data_folder'],
                                               train=False,
                                               transform=test_transform,
                                               download=True)

            config["ood_dataset_names"] = ["LSUN", "TIN", "SVHN", "CIFAR", "iSUN"]

            svhn_ood_test_data = dset.SVHN(config['data_folder'], split='test',
                                           transform=test_transform,
                                           download=True)

             # download if necessary
            if not (os.path.exists(config['data_folder'] + '/LSUN')):
                urllib.request.urlretrieve('https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1',config['data_folder'] + '/LSUN_resize.tar.gz')
                tar = tarfile.open(config['data_folder'] + '/LSUN_resize.tar.gz', "r:gz")
                tar.extractall(path=config['data_folder'] + '/LSUN')
                tar.close()

            lsun_ood_test_data = dset.ImageFolder(config['data_folder'] + '/LSUN', 
                                                        transform=test_transform)

             # download if necessary
            if not (os.path.exists(config['data_folder'] + '/TIN')):
                urllib.request.urlretrieve('https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz?dl=1', config['data_folder'] + '/Imagenet_resize.tar.gz')
                tar = tarfile.open(config['data_folder'] + '/Imagenet_resize.tar.gz', "r:gz")
                tar.extractall(path=config['data_folder'] + '/TIN')
                tar.close()

            tin_ood_test_data = dset.ImageFolder(config['data_folder'] + '/TIN', 
                                                        transform=test_transform)
            
            if not (os.path.exists(config['data_folder'] + '/iSUN')):
                urllib.request.urlretrieve('https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz?dl=1', config['data_folder'] + '/iSUN.tar.gz')
                tar = tarfile.open(config['data_folder'] + '/iSUN.tar.gz', "r:gz")
                tar.extractall(path=config['data_folder'] + '/iSUN')
                tar.close()
            
            iSUN_ood_test_data = dset.ImageFolder(config['data_folder'] + '/iSUN',
                                                        transform=test_transform)

            ood_test_datasets = [lsun_ood_test_data, tin_ood_test_data,
                                                     svhn_ood_test_data, 
                                                     c_ood_test_data,
                                                     iSUN_ood_test_data]
            
            ood_test_loader_list = []
            for ood_dataset in ood_test_datasets:
                ood_test_loader_list += [torch.utils.data.DataLoader(
                    ood_dataset, batch_size=config['test_batch_size'],
                    shuffle=False, num_workers=config['num_workers'],
                    pin_memory=True)]

        elif config['problem'] == "oodSUBCIFAR100":
            raise Exception('Currently unsupported.')

        else:
            raise Exception('Please configure CIFAR problem correctly.')

    else:
        raise Exception('Unknown problem.')

    return train_loader, test_loader, \
        ood_test_loader_list
