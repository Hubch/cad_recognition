from PIL import Image

def crop_image(image_path):
    # 打开原始图片
    image = Image.open(image_path)
    width, height = image.size

    # 计算裁剪后的宽度和高度
    crop_width = width // 3
    crop_height = height // 3

    images = []
    for i in range(3):
        for j in range(3):
            # 计算每个裁剪区域的坐标
            left = j * crop_width
            upper = i * crop_height
            right = left + crop_width
            lower = upper + crop_height

            # 裁剪图像
            cropped_image = image.crop((left, upper, right, lower))
            images.append(cropped_image)

    return images




if __name__ == "__main__":
    
    image_path = 'I:/company project/cad_recognition/backend/run_logs/output_page.png'
    # 裁剪图片
    cropped_images = crop_image(image_path)
    # 保存裁剪后的图片
    for i, image in enumerate(cropped_images):
        output_path = f'I:/company project/cad_recognition/backend/run_logs/output_{i}.png'
        image.save(output_path)
