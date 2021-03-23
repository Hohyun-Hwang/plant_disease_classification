import os  # for reading directory information
from glob import glob  # for open lots of file
from PIL import Image  # for open image file
from torch.utils.data.dataset import Dataset  # for custom dataloader
import torch
"""

* Folder structure
=========
-Dataset
|--train
|---Apple___Apple_scab
|---Apple___Black_rot
|---Apple___Cedar_apple_rust
|---Apple___healthy
|---Blueberry___healthy
|---...
|--test
|---Apple___Apple_scab
|---Apple___Black_rot
|---Apple___Cedar_apple_rust
|---Apple___healthy
|---Blueberry___healthy
|---...
=========
"""


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
                      'Early_blight',
                      'Septoria_leaf_spot',
                      'Tomato_mosaic_virus',
                      'Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Spider_mites Two-spotted_spider_mite',
                      'Cercospora_leaf_spot Gray_leaf_spot',
                      'Northern_Leaf_Blight',
                      'Bacterial_spot',
                      'Powdery_mildew',
                      'Common_rust_',
                      'Cedar_apple_rust',
                      'Leaf_scorch',
                      'Late_blight',
                      'Tomato_Yellow_Leaf_Curl_Virus',
                      'Target_Spot',
                      'Esca_(Black_Measles)',
                      'Leaf_Mold',
                      'Apple_scab',
                      'Black_rot',
                      'Haunglongbing_(Citrus_greening)']

        # label_list = ['healthy',
        #               'Cedar_apple_rust',
        #               'Apple_scab',
        #               'Black_rot']

        # label_torch = [0 for x in range(0,21)]
        for label_idx, label_file in enumerate(label_list):
            if label_name == label_file:
                # import ipdb;ipdb.set_trace()
                # label_torch[label_idx] = 1
                # print(label_torch)
                label_torch = label_idx



        return img, torch.tensor(label_torch)


if __name__ == "__main__":
    from torchvision.transforms import transforms
    from torch.utils.data.dataloader import DataLoader

    dir = "dataset_itr2/train"
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    trainset = plant_image(data_dir=dir, transform=transform)
    train_loader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)
    # print(len(train_loader.dataset))
    for img, label in train_loader:
        print(img)
        print(label)

        import sys;
        sys.exit(0)
