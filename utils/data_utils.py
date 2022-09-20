import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=1, root='../data'):
    img_size = 224

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=transform_test)

    elif dataset == 'tiny':
        trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val', transform=transform_test)

    elif dataset == 'imagenet':
        trainset = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/ImageNet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/ImageNet/val', transform=transform_test)

    else:
        raise NotImplementedError

    train_loader = DataLoader(trainset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, test_loader