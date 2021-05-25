import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
#from utils import download_gdrive
import sys

def load_cifar10(n_examples, data_dir='./data', training_set=False, device='cuda'):
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=training_set, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_examples].to(device)
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_examples].to(device)

    return x_test, y_test



def load_imagenet(n_examples, training_set=False, return_loader=False):
    IMAGENET_SL = 224
    if not training_set:
        IMAGENET_PATH = "/home/scratch/datasets/imagenet/val"
        if not os.path.exists(IMAGENET_PATH):
            IMAGENET_PATH = "/scratch/maksym/imagenet/val_orig"
    else:
        IMAGENET_PATH = "/home/scratch/datasets/imagenet/train"
    imagenet = datasets.ImageFolder(IMAGENET_PATH,
                           transforms.Compose([
                               transforms.Resize(IMAGENET_SL + 32),
                               transforms.CenterCrop(IMAGENET_SL),
                               transforms.ToTensor()
                           ]))
    torch.manual_seed(0)

    test_loader = data.DataLoader(imagenet, batch_size=n_examples, shuffle=True, num_workers=30)
    
    if return_loader:
        from robustness.tools import helpers
        return helpers.DataPrefetcher(test_loader)
    testiter = iter(test_loader)
    x_test, y_test = next(testiter)
    
    return x_test, y_test

#

def load_anydataset(args, device='cuda'):
    if args.dataset == 'cifar10':
        x_test, y_test = load_cifar10(args.n_ex, args.data_dir,
            args.training_set, device=device)
        #x_test = x_test.contiguous()
    elif args.dataset == 'imagenet':
        x_test, y_test = load_imagenet(args.n_ex, args.training_set)
    elif args.dataset == 'cifar100':
        x_test, y_test = load_cifar100(args.n_ex, '/home/scratch/datasets/CIFAR100',
            args.training_set, device=device)
    elif args.dataset == 'imagenet100':
        x_test, y_test = load_imaget100(args.n_ex)
    
    return x_test, y_test

def load_cifar100(n_examples, data_dir='/home/scratch/datasets/CIFAR100', training_set=False, device='cuda'):
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR100(root=data_dir, train=training_set, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_examples].to(device)
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_examples].to(device)

    return x_test, y_test



    
    
# data loaders training
def load_cifar10_train(args, only_train=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    root = args.data_dir + '' #'/home/EnResNet/WideResNet34-10/data/'
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        root, train=True, transform=train_transform, download=True)
    if not only_train:
      test_dataset = datasets.CIFAR10(
          root, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    if not only_train:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
    else:
        test_loader = ()
    
    return train_loader, test_loader

def load_imagenet_train(args):
    from robustness.datasets import DATASETS
    from robustness.tools import helpers
    dataset = DATASETS['imagenet'](args.data_dir) #'/home/scratch/datasets/imagenet'
    
    
    train_loader, val_loader = dataset.make_loaders(30,
                    args.batch_size, data_aug=True)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    return train_loader, val_loader

if __name__ == '__main__':
    #x_test, y_test = load_cifar10c(100, corruptions=['fog'])
    x_test, y_test = load_imagenet100(100)
    print(x_test.shape, x_test.max(), x_test.min())


