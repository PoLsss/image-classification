import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import os
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        mode = 'train' if train else 'valid'
        self.root = os.path.join(root, mode)
        # print(os.listdir(self.root))
        self.categories = ['cats', 'dogs']

        for indx, category in enumerate(self.categories):
            data_path = os.path.join(self.root, category)
            for file_name in os.listdir(data_path):
                file_path = os.path.join(data_path, file_name)
                self.images.append(file_path)
                self.labels.append(indx)
        # print(self.image[1])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[index]
        return image, label

if __name__ == '__main__':
    root = 'data/'
    transform = Compose([
        Resize((512,512)),
        ToTensor(),
    ])

    dataset = CatDogDataset(root=root, train=True, transform=transform)
    image, label = dataset.__getitem__(1234)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

    for images, labels in dataloader:
        print(images.shape)
        print(labels)
        break
