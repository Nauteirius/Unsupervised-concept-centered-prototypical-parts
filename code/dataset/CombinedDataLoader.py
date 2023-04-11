import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import torch
import numpy as np


class CDL(Dataset):
    base_folder = 'CUB_200_2011'

    # root - root folder
    # train - True - train set, False - test set, None - whole set
    # images - list of names of folders with images to use
    # num_classes = number of classes to load
    def __init__(self, root, train=True, transform=None, images=['images'], num_classes=200):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.images = images
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        self.data = data[data.target <= num_classes]

        if self.train is None:
            return
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data) * len(self.images)

    def __getitem__(self, idx):
        out_idx = idx // len(self.data)
        in_idx = idx % len(self.data)
        sample = self.data.iloc[in_idx]
        path = os.path.join(self.root, self.base_folder, self.images[out_idx], sample.filepath)
        target = sample.target - 1
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
