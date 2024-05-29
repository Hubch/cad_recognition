import copy
import json
from typing import List
from openai.types.chat import ChatCompletionMessageParam
import base64
import json
from PIL import Image
import numpy as np
from io import BytesIO
from utils.polt import Annotator,colors

def pprint_prompt(prompt_messages: List[ChatCompletionMessageParam]):
    print(json.dumps(truncate_data_strings(prompt_messages), indent=4))


def truncate_data_strings(data: List[ChatCompletionMessageParam]):  # type: ignore
    # Deep clone the data to avoid modifying the original object
    cloned_data = copy.deepcopy(data)

    if isinstance(cloned_data, dict):
        for key, value in cloned_data.items():  # type: ignore
            # Recursively call the function if the value is a dictionary or a list
            if isinstance(value, (dict, list)):
                cloned_data[key] = truncate_data_strings(value)  # type: ignore
            # Truncate the string if it it's long and add ellipsis and length
            elif isinstance(value, str):
                cloned_data[key] = value[:40]  # type: ignore
                if len(value) > 40:
                    cloned_data[key] += "..." + f" ({len(value)} chars)"  # type: ignore

    elif isinstance(cloned_data, list):  # type: ignore
        # Process each item in the list
        cloned_data = [truncate_data_strings(item) for item in cloned_data]  # type: ignore

    return cloned_data  # type: ignore

def extract_json_from_text(text):
    # 在文本中查找 JSON 字符串
    start = text.find('{')
    end = text.rfind('}')
    json_str = text[start:end+1]

    # 解析 JSON 字符串
    try:
        json_data = json.loads(json_str)
        print(f"json_data:{json_data}")
        return json_data
    except json.JSONDecodeError:
        print("无法解析 JSON 数据")


def base64_to_pil(base64_string):
    # 将base64字符串解码为图像
    if base64_string.find("data:image/jpeg;base64") != -1:
        base64_string = base64_string.split(",")[1]
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
    except Exception as e :
        print(e)
    return image

def pil_to_base64(image):
    # 将图像转换为base64字符串
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_image

def convertBase64(image):
    # 将NumPy数组转换为PIL图像
    if isinstance(image, np.ndarray):
        # 如果是NumPy数组，使用PIL的fromarray函数将其转换为PIL图像
        if len(image.shape) == 4:
            image = image.squeeze(0).transpose((1, 2,0))
            print(f"convert:{image.shape}")
        # 检查数组数据类型并转换为 uint8
        if image.dtype != np.uint8:
            # 确保值在0-255范围内
            image = np.clip(image, 0, 1)
            # 转换为 uint8 类型
            image = (image * 255).astype(np.uint8)
            
        image = Image.fromarray(image)
    # 将 RGBA 图像转换为 RGB 图像
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # 创建一个字节流对象
    buffered = BytesIO()
    # 将PIL图像保存到字节流中，格式为'JPEG'或'PNG'
    image.save(buffered, format="JPEG")
    # 获取字节流的bytes
    byte_data = buffered.getvalue()
    # 对字节数据进行Base64编码
    base64_encoded = base64.b64encode(byte_data)
        # 将编码后的bytes转换为字符串
    base64_string = base64_encoded.decode('utf-8')
    return base64_string

def draw_box(images, result):
    base64_images = []
    differences = result.get("differences")
    boxes = differences.get("images")
    if result:
        for index,image_result in enumerate(boxes) :
          
            boxes = image_result['boxes']
            image_base64 = images[index]
            image_pil = base64_to_pil(image_base64)
            image_pil = np.ascontiguousarray(image_pil)
            annotator = Annotator(image_pil, example=str("闸阀"),font_size=12,font="msyh.ttc",pil=True)
            color = colors(index, True)
            for box in boxes:
                bbox = box['bbox']
                label = box['label']
                annotator.box_label(box= bbox, label= label, color=color,rotated = False)
            # 将图像转换回PIL格式
            image_pil = annotator.result()
            # 将图像转换为base64字符串
            base64_image = convertBase64(image_pil)
            base64_images.append(base64_image)
    else:
        return images
    return base64_images