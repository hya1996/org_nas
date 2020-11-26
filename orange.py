from torch.utils.data import Dataset
import os
import PIL.Image as Image
import numpy as np


class ORANGE(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.images = self._get_images()

    def _get_images(self):
        images = []
        for index, cls in enumerate(self.classes):
            image_name = os.listdir(os.path.join(self.root, cls))
            for img in image_name:
                sample = (os.path.join(self.root, cls, img), index)
                if sample[0].endswith(".jpg") or sample[0].endswith(".JPG"):
                    images.append(sample)

        return images

    def __getitem__(self, index):
        img = Image.open(self.images[index][0])
        target = self.images[index][1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


class Subset(Dataset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)


def dataset_split(dataset, test_ratio, random=False, train_transform=None, test_transform=None, target_transform=None):
    num_dataset = len(dataset)
    num_test = int(num_dataset * test_ratio)
    num_train = int(num_dataset - num_test)
    indices = np.arange(num_dataset)
    if random:
        indices = np.random.permutation(indices)
    indices_train = indices[:num_train]
    indices_test = indices[num_train:]
    dataset_train = Subset(dataset, indices_train, transform=train_transform)
    dataset_test = Subset(dataset, indices_test, transform=test_transform)

    return dataset_train, dataset_test
