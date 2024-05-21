import os
import cv2
import numpy as np

# 输入文件夹路径
input_dir = 'input/cropped_images'

# 遍历文件夹，获取所有图像的尺寸
image_sizes = []
for class_dir in os.listdir(input_dir):
    class_dir_path = os.path.join(input_dir, class_dir)
    for image_file in os.listdir(class_dir_path):
        image_path = os.path.join(class_dir_path, image_file)
        image = cv2.imread(image_path)
        image_sizes.append(image.shape[:2])

# 将图像尺寸转换为NumPy数组
image_sizes = np.array(image_sizes)

# 计算裁剪的小图像的平均尺寸
mean_height = np.mean(image_sizes[:, 0])
mean_width = np.mean(image_sizes[:, 1])

print(f'Mean Height: {mean_height}, Mean Width: {mean_width}')

# 选择一个合适的输入尺寸
input_height = int(mean_height)
input_width = int(mean_width)

print(f'Selected Input Height: {input_height}, Selected Input Width: {input_width}')
