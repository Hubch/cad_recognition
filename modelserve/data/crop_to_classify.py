import os
import cv2

# 输入路径
images_dir = 'input/images'
labels_dir = 'input/labels'
output_dir = 'input/cropped_images'

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历标签文件
for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    image_path = os.path.join(images_dir, label_file.replace('.txt', '.png'))

    # 读取图像
    image = cv2.imread(image_path)

    # 读取标签信息
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 遍历每个目标
    for line in lines:
        class_id, center_x, center_y, width, height = map(float, line.strip().split())

        # 计算目标左上角坐标
        x1 = int((center_x - width / 2) * image.shape[1])
        y1 = int((center_y - height / 2) * image.shape[0])
        # 计算目标右下角坐标
        x2 = int((center_x + width / 2) * image.shape[1])
        y2 = int((center_y + height / 2) * image.shape[0])

        # 裁剪目标图像
        cropped_image = image[y1:y2, x1:x2]

        # 创建目标类别文件夹
        class_output_dir = os.path.join(output_dir, f'class_{int(class_id)}')
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # 保存裁剪后的图像
        output_image_path = os.path.join(class_output_dir, f'{label_file.replace(".txt", "")}_{x1}_{y1}_{x2}_{y2}.png')
        cv2.imwrite(output_image_path, cropped_image)

        # 记录裁剪后的图像尺寸
        print(f'Saved cropped image: {output_image_path}, Size: {cropped_image.shape}')
