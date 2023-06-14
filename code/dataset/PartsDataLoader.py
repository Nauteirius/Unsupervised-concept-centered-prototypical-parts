import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms



class PDL(Dataset):
    base_folder = 'augmented'

    def __init__(self, root, train=True, transform=None, images=['images'], num_classes=200):
        self.root = os.path.expanduser(root) 
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.images = images
        self.folders = ['IMG', 'PART_0', 'PART_1', 'PART_2', 'PART_3']

        images = pd.read_csv(os.path.join(self.root, self.base_folder, 'images_cleaned.txt'), sep=' ',
                             names=['img_id', 'filepath'])

        image_class_labels = pd.read_csv(os.path.join(self.root, self.base_folder, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, self.base_folder, 'train_test_split.txt'),
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

        # check if classes match
        # print(self.data.to_string())

    def __len__(self):
        return len(self.data) * len(self.images)

    def __getitem__(self, idx):
        out_idx = idx // len(self.data)
        in_idx = idx % len(self.data)
        sample = self.data.iloc[in_idx]
        filename = sample.filepath.split('/')[-1] 
        data = {}
        for folder in self.folders:
            path = os.path.join(self.root, self.base_folder, self.images[out_idx], folder, filename)
            img = self.loader(path)
            data[folder] = img

        target = sample.target - 1
        if self.transform is not None:
            for folder, img in data.items():
                data[folder] = self.transform(img)
        return data, target

def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        ])

    dataset = PDL(root='../', train=False, transform=transform, images=['images'], num_classes=20)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (data, labels) in enumerate(data_loader):
        img = data['IMG'][0]
        parts = data['PART_0'][0], data['PART_1'][0], data['PART_2'][0], data['PART_3'][0]
        # print images using matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        npimg = img.numpy()
        plt.subplot(1, 5, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f'Original Image: {labels[0]}')
        # turn off axis
        plt.axis('off')
        for i, part in enumerate(parts):
            npimg = part.numpy()
            plt.subplot(1, 5, i+2)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            # turn off axis
            plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
