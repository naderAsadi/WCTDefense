import torch
import torchvision

def get_loader(dataset, image_size, batch_size, dataroot, train):
    """
    :param dataset:
    :param image_size: 
    :param batch_size:
    :param dataroot:
    :param train:
    :return:
    """
    if dataset == 'mnist':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(dataroot, train=train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize(image_size),
                                           torchvision.transforms.ToTensor(),
                                       ])),
            batch_size=batch_size, shuffle=False)
    elif dataset == 'emnist':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.EMNIST(dataroot, split='fonts', train=train, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.ToTensor()
                                        ])),
            batch_size=batch_size, shuffle=False
        )
    elif dataset == 'svhn':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(dataroot, train=train , download=True,
                                       transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.ToTensor(),
                                        ])),
            batch_size=batch_size, shuffle=False)
    return loader