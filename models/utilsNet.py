# utils functions for networks model configurations
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def transform(num_output_channels=3, normalize=True):
    tGrayscale = transforms.Grayscale(num_output_channels=num_output_channels)

    if normalize:
        data_transforms = transforms.Compose(
            [
                tGrayscale,
                transforms.ToTensor(),
                # Normalize input channels using mean values and standard deviations of ImageNet.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                tGrayscale,
                transforms.ToTensor()
            ]
        )

    return data_transforms


def createTrainValDataLoaders(DATA_DIR, batch_size, num_output_channels=3, normalize=True):
    # create dataloaders
    data_transforms = transform(num_output_channels=num_output_channels, normalize=normalize)

    train_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms)
    train_loader = DataLoader(dataset=train_image_datasets, shuffle=True, batch_size=batch_size)

    val_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), data_transforms)
    val_loader = DataLoader(dataset=val_image_datasets, shuffle=False, batch_size=batch_size)

    # merge train & val data loaders
    dataloaders = {'train': train_loader, 'validation': val_loader}
    dataset_sizes = {'train': len(train_image_datasets), 'validation': len(val_image_datasets)}

    return dataloaders, dataset_sizes


def createTestDataLoaders(DATA_DIR, batch_size=1, num_output_channels=3, normalize=True):
    # create dataloaders
    data_transforms = transform(num_output_channels=num_output_channels, normalize=normalize)
    # create dataloaders
    test_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), data_transforms)
    test_loader = DataLoader(dataset=test_image_datasets, shuffle=False, batch_size=batch_size)

    return test_loader