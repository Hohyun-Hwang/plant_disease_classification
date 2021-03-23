import os  # for reading directory information
from glob import glob  # for open lots of file
from PIL import Image  # for open image file
from torch.utils.data.dataset import Dataset  # for custom dataloader
import torch

class plant_image(Dataset):

    def __init__(self, data_dir, transform):
        self.data_dir = glob(data_dir + "/**/*.JPG")
        self.transform = transform

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        filepath = self.data_dir[idx]
        label_name = filepath.split('/')[-2].split("___")[1]
        img = Image.open(filepath)

        if self.transform:
            img = self.transform(img)


        label_list = ['healthy',
                      'Cedar_apple_rust',
                      'Apple_scab',
                      'Black_rot']

        for label_idx, label_file in enumerate(label_list):
            if label_name == label_file:
                label_torch = label_idx



        return img, torch.tensor(label_torch)


if __name__ == "__main__":
    from torchvision.transforms import transforms
    from torch.utils.data.dataloader import DataLoader

    dir = "apple/train"
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    trainset = plant_image(data_dir=dir, transform=transform)
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    for img, label in train_loader:
        print(img)
        print(label)

        import sys;
        sys.exit(0)
