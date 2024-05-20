import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
import torch
import torch.nn as nn
from tqdm import tqdm

# 输入文件夹路径
input_dir = 'cropped_images'

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet模型
resnet = models.resnet18(pretrained=True)
# 去除最后一层全连接层
resnet = nn.Sequential(*list(resnet.children())[:-1])
# 设置为评估模式
resnet.eval()
# 移动模型到设备
resnet = resnet.to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # 设定输入尺寸
    transforms.ToTensor(),
])

# 定义类别文件夹路径
class_dirs = [os.path.join(input_dir, class_dir) for class_dir in os.listdir(input_dir)]

# 定义类别特征向量字典
class_features = {}

# 提取特征向量
for class_dir in tqdm(class_dirs, desc='Extracting Features'):
    class_name = os.path.basename(class_dir)
    class_features[class_name] = []
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        # 读取图像
        image = cv2.imread(image_path)
        # 图像预处理
        image_tensor = transform(image).unsqueeze(0).to(device)
        # 提取特征向量
        with torch.no_grad():
            feature = resnet(image_tensor).squeeze().cpu().numpy()
        # 将特征向量添加到类别特征列表中
        class_features[class_name].append(feature)

# 计算每个类别的平均特征向量
for class_name, features in class_features.items():
    class_features[class_name] = np.mean(features, axis=0)

# 计算类别之间的余弦相似度
for class_name1, features1 in class_features.items():
    for class_name2, features2 in class_features.items():
        if class_name1 != class_name2:
            similarity = cosine_similarity([features1], [features2])[0][0]
            print(f"Similarity between {class_name1} and {class_name2}: {similarity}")
