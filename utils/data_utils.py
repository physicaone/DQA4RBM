from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(dataset_name, n_visible):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST('../DQA2/data', train=True, download=False, transform=transform)
    elif dataset_name == 'fMNIST':
        dataset = datasets.FashionMNIST('../DQA2/data', train=True, download=False, transform=transform)
    elif dataset_name == 'kMNIST':
        dataset = datasets.KMNIST('../DQA2/data', train=True, download=False, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    train_dataset, val_dataset = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False)
    return train_loader, val_loader
