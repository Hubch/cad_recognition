import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image

# 构建简单的自定义数据集
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                samples.append((img_path, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx

# 构建三元组数据集
class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor, label = self.dataset[idx]
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.randint(0, len(self.dataset))
        positive, _ = self.dataset[positive_idx]
        
        negative_idx = idx
        while negative_idx == idx or self.dataset[negative_idx][1] == label:
            negative_idx = np.random.randint(0, len(self.dataset))
        negative, _ = self.dataset[negative_idx]

        return anchor, positive, negative

# 创建模型
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base_model = base_model
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        features = self.feature_extractor(x).squeeze()
        return features

# 定义训练函数
def train_triplet_loss(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# 定义三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, 2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)

# 设置随机种子
torch.manual_seed(0)

# 定义训练参数
batch_size = 32
lr = 0.001
num_epochs = 10
margin = 1.0

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 根据之前计算的最佳尺寸进行调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = CustomDataset(root_dir='cropped_images', transform=transform)
triplet_dataset = TripletDataset(train_dataset)
train_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

# 创建模型和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model = FeatureExtractor(base_model).to(device)
criterion = TripletLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
print("start to train")
train_triplet_loss(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

# 保存模型
torch.save(model.state_dict(), 'feature_extractor_mobilenetv2.pth')
