import os
import random
import numpy as np
from PIL import Image, ImageStat
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import json 
import cv2

# 数据预处理和加载
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)
    
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.RandomCrop(32, padding=2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    CLAHETransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])
transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])


class CombinedDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_map_path='./weights/class_map.json', num_samples_per_class=5):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.data = []
        self.labels = []
        self.num_samples_per_class = num_samples_per_class
        self.class_map_path = class_map_path
        
        if "train" in root_dir:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
            self.save_class_mapping()
        else:
            self.class_to_idx = self.load_class_mapping(class_map_path)
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
                            
        for class_dir in self.classes:
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if not img_name.endswith("png"):
                        print(f"filename error: {img_name}")
                        continue
                    img_path = os.path.join(class_path, img_name)
                    if self.is_valid_image(img_path):
                        self.data.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir])
        
        # self.data, self.labels = self.balance_dataset()
        self.sample_weights = self.compute_sample_weights()
        
        if "train" in root_dir:
            self.class_weights = self.compute_class_weights()
            self.save_class_weights()

    def __len__(self):
        return len(self.data) * self.num_samples_per_class

    def __getitem__(self, idx):
        base_idx = idx // self.num_samples_per_class
        anchor_path = self.data[base_idx]
        anchor_label = self.labels[base_idx]
        anchor_img = Image.open(anchor_path).convert('L')

        positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == anchor_label and i != idx])
        positive_path = self.data[positive_idx]
        positive_label = self.labels[positive_idx]
        positive_img = Image.open(positive_path).convert('L')

        negative_idx = random.choice([i for i, label in enumerate(self.labels) if label != anchor_label])
        negative_path = self.data[negative_idx]
        negative_label = self.labels[negative_idx]
        negative_img = Image.open(negative_path).convert('L')
        

        _anchor_img = self.transform(anchor_img)
        contrastive_img = self.transform(anchor_img)
        positive_img = self.transform(positive_img)
        negative_img = self.transform(negative_img)
        return _anchor_img, contrastive_img, positive_img, negative_img, anchor_label, positive_label, negative_label
            

    def get_class_name(self, idx):
        return self.idx_to_class.get(idx, "Unknown")

    def save_class_mapping(self):
        with open(self.class_map_path, 'w') as f:
            json.dump(self.class_to_idx, f)

    @staticmethod
    def load_class_mapping(class_map_path):
        with open(class_map_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def is_valid_image(img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert('L')
                stat = ImageStat.Stat(img)
                if stat.mean[0] < 5:  
                    print(f"Invalid image (almost black): {img_path}")
                    return False
                if stat.mean[0] > 250:  
                    print(f"Invalid image (almost white): {img_path}")
                    return False
                if stat.stddev[0] < 1:  
                    print(f"Invalid image (low contrast): {img_path}")
                    return False
                return True
        except Exception as e:
            print(f"Invalid image file: {img_path}, error: {e}")
            return False
    
    def balance_dataset(self):
        data_res = np.array([np.array(Image.open(d).convert('L').resize((32, 32))).flatten() for d in self.data])
        labels_res = np.array(self.labels)

        smote = SMOTE(sampling_strategy='auto')
        under = RandomUnderSampler(sampling_strategy='auto')
        steps = [('o', smote), ('u', under)]
        pipeline = Pipeline(steps=steps)

        data_res, labels_res = pipeline.fit_resample(data_res, labels_res)

        return data_res, labels_res
        
    def compute_sample_weights(self):
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels]
        
        # Normalize sample weights
        sample_weights = sample_weights / sample_weights.sum() * total_samples
        return sample_weights
    
    def compute_class_weights(self):
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # Normalize class weights
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        return class_weights.tolist()

    def save_class_weights(self, weight_path='./weights/class_weights.json'):
        with open(weight_path, 'w') as f:
            json.dump(self.class_weights, f)
    
    def load_class_weight(self, weights_path='./weights/class_weights.json'):
        with open(weights_path, 'r') as f:
            return json.load(f)
    
        
def get_data_loaders(root_dir='../datas/cropped_images', mode='train', batch_size=256):
    data_dir = os.path.join(root_dir, mode)
    dataset = CombinedDataset(data_dir, transform_train if mode == 'train' else transform_val)
    
    if mode == 'train':
        # sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset.sample_weights))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    return dataloader, len(dataset.classes)