import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 数据预处理和加载
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomCrop(32, padding=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])
transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.classes = os.listdir(root_dir)
        self.data = []
        self.labels = []

        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if not img_name.endswith("png"):
                        print(f"filename error: {img_name}")
                        continue
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path = self.data[idx]
        anchor_label = self.labels[idx]
        anchor_img = Image.open(anchor_path).convert('L')
        if self.transform:
            anchor_img = self.transform(anchor_img)

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == anchor_label])
        positive_path = self.data[positive_idx]
        positive_label = self.labels[positive_idx]
        positive_img = Image.open(positive_path).convert('L')
        if self.transform:
            positive_img = self.transform(positive_img)

        negative_idx = random.choice([i for i, label in enumerate(self.labels) if label != anchor_label])
        negative_path = self.data[negative_idx]
        negative_label = self.labels[negative_idx]
        negative_img = Image.open(negative_path).convert('L')
        if self.transform:
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label, positive_label, negative_label
    

def get_data_loaders(root_dir='../datas/cropped_images', mode='train', batch_size=256):
    data_dir = os.path.join(root_dir, mode)
    if mode == 'train':
        dataset = TripletDataset(data_dir, transform_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    else:
        dataset = TripletDataset(data_dir, transform_val)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    return dataloader