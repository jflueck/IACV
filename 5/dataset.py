import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels if labels is not None else np.zeros(len(images))
        self.unique_labels = list(set(self.labels))
        self.num_classes = len(self.unique_labels)
        self.image_size = images.shape[1:]
        self.data_size = (self.image_size[2], self.image_size[0], self.image_size[1])
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image.astype('uint8'))  # Convert numpy array to PIL Image
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_datasets_from_h5(h5_path, train_transform=None, test_transform=None):
    with h5py.File(h5_path, "r") as f:
        train_images = f["train_images"][:]
        train_labels = f["train_labels"][:]
        val_images = f["val_images"][:]
        val_labels = f["val_labels"][:]
        test_images = f["test_images"][:]
        test_labels = f["test_labels"][:]
    train_dataset = ImageDataset(train_images, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_images, val_labels, transform=None)
    test_dataset = ImageDataset(test_images, test_labels, transform=test_transform)
    return train_dataset, val_dataset, test_dataset

def get_test_loader(test_h5_path):
    with h5py.File(test_h5_path, "r") as f:
        images = f["images"][:]
    test_dataset = ImageDataset(images)
    return DataLoader(test_dataset, batch_size=50, shuffle=False)


def get_loaders_from_datasets(train_dataset, val_dataset, test_dataset, batch_size=50):
    """
    Takes the datasets and returns the train, validation, and test loaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader