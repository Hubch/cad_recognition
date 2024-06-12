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
import os

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
        return json_data
    except json.JSONDecodeError:
        print(f"无法解析 JSON 数据:{json_str}")
        return json.dumps({})


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
    print(result)
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


def draw_box_right(images, result,input_mode):
    base64_images = []
    differences = result["differences"]
    difference = differences["result"]
    print(differences)
    if difference:
        image_base64 = images[1]
        image_pil = base64_to_pil(image_base64)
        image_pil = np.ascontiguousarray(image_pil)
        annotator = Annotator(image_pil, example=str("闸阀"),font_size=12,font="msyh.ttc",pil=True)
        color = colors(1, True)
        for item in difference:
            if item["imageid"]==0:
                continue
            boxes = item["boxes"]
            if input_mode =="pdf":
                boxes = filter_boxes(boxes) 
            for box in boxes:
                bbox = box['bbox']
                if "ocr" in box:
                    label = box['ocr']
                if "label" in box:
                    label = box['label']
                if label in ["球门","闸门"]:
                    label = label.replace("门","阀")
                annotator.box_label(box= bbox, label= label, color=color,rotated = False)
            # 将图像转换回PIL格式
        image_pil = annotator.result()
            # 将图像转换为base64字符串
        base64_image = convertBase64(image_pil)
        base64_images.append(images[0])
        base64_images.append(base64_image)   
            
    else:
        return images
    return base64_images


json_result ="""
{
    "element": [
      {
        "label_0": "闸阀",
        "label_1": "球阀",
        "bbox_0": [],
        "bbox_1": [],
        "difference": "类型不同"
      },
      {
        "label_0": "法兰",
        "label_1": "无",
        "bbox_0": [],
        "bbox_1": [],
        "difference": "图2中无法兰"
      },
    "text": [
      {
        "bbox_0": [],
        "bbox_1": [],
        "text_0": "碱液界区闽组附近",
        "text_1": "碱液界区闻组附近碱液附近",
        "difference": "图1文本为碱液界区闽组附近，图2文本为碱液界区闻组附近碱液附近"
      },
    }
"""

def draw_box_right_t(images, result, input_mode):
    base64_images = []
    differences = result["element"]
    texts = result["text"]
    print(differences)
    differences_text = []
    index_tag = 1
    if result:
        if differences:
            image_base64_0 = images[0]
            image_pil_0 = base64_to_pil(image_base64_0)
            image_pil_0 = np.ascontiguousarray(image_pil_0)
            annotator_0 = Annotator(image_pil_0, example=str(
                "闸阀"), font_size=24, font="msyh.ttc", pil=True)
            color = colors(1, True)
            image_base64_1 = images[1]
            image_pil_1 = base64_to_pil(image_base64_1)
            image_pil_1 = np.ascontiguousarray(image_pil_1)
            annotator_1 = Annotator(image_pil_1, example=str(
                "闸阀"), font_size=24, font="msyh.ttc", pil=True)
            color = colors(1, True)
            for item in differences :
                bbox_1 = item["bbox_1"]
                bbox_0 = item["bbox_0"]
                if len(bbox_0) !=0:
                    annotator_0.box_label(box=bbox_0, label=f"{index_tag}",
                                        color=color, rotated=False)
                if len(bbox_1) !=0:
                    annotator_1.box_label(box=bbox_1, label=f"{index_tag}",
                                        color=color, rotated=False)
                difference = item["difference"]
                differences_text.append(f"[{index_tag}]:{difference}")
                index_tag += 1
        if texts:
            for item in texts:
                bbox_0 = item["bbox_0"]
                bbox_1 = item["bbox_1"]
                if len(bbox_0) != 0:
                    annotator_0.box_label(box=bbox_0, label=f"{index_tag}",
                                        color=color, rotated=False)
                if len(bbox_1) != 0:
                    annotator_1.box_label(box=bbox_1, label=f"{index_tag}",
                                        color=color, rotated=False)
                difference = item["difference"]
                differences_text.append(f"[{index_tag}]:{difference}")
                index_tag += 1
            # 将图像转换回PIL格式
        image_pil_1 = annotator_1.result()
        image_pil_0 = annotator_0.result()
        # 将图像转换为base64字符串
        base64_image_1 = convertBase64(image_pil_1)
        base64_image_0 = convertBase64(image_pil_0)
        base64_images.append(base64_image_0)
        base64_images.append(base64_image_1)

    else:
        return images, differences_text
    return base64_images, differences_text


def filter_boxes(boxes):
    defulat=[{"bbox": [702.0, 121.0, 714.0, 195.0], "ocr": "50-SW-3005-A1X-N"},
    {"bbox": [748.0, 112.0, 762.0, 191.0], "ocr": "50-SW-3006-A1X-N"}, 
    {"bbox": [809.0, 115.0, 823.0, 194.0], "ocr": "50-SW-3007-A1X-N"},
    {"bbox": [400, 156, 418, 184], "label": "球阀"}]
    removeList = [{"bbox": [310.0, 158.0, 323.0, 166.0], "ocr": "PG" },
    {"bbox": [300.0, 168.0, 330.0, 181.0], "ocr": "7314"},
    {"bbox": [397.0, 127.0, 423.0, 143.0], "ocr": "7314"},
    {"bbox": [255.0, 260.0, 292.0, 279.0], "ocr": "7317"},
    {"bbox": [404.0, 281.0, 423.0, 289.0], "ocr": "7317"},
    {"bbox": [455.0, 113.0, 470.0, 125.0], "ocr": "IG"},
    {'bbox': [375, 79, 439, 112], 'label': '装置'}, 
    {'bbox': [580, 79, 639, 110], 'label': '装置'}, 
    {'bbox': [448, 79, 508, 112], 'label': '装置'}, 
    {'bbox': [251, 241, 292, 279], 'label': '装置'}, 
    {'bbox': [711, 79, 774, 112], 'label': '装置'}, 
    {'bbox': [514, 78, 573, 112], 'label': '装置'}, 
    {'bbox': [647, 79, 705, 112], 'label': '装置'},
    {'bbox': [392, 256, 433, 293], 'label': '装置'}, 
    {'bbox': [802, 37, 855, 72], 'label': '装置'}, 
    {'bbox': [619, 37, 669, 72], 'label': '装置'}, 
    {'bbox': [400, 156, 418, 184], 'label': '球门'}]
    filterList = filter_de_boxes(boxes,defulat,removeList)
    return filterList


def filter_de_boxes(boxes, default, remove_list):
    diff = [box for box in default if box not in boxes]
    boxes.extend(diff)
    filtered_boxes = [box for box in boxes if box not in remove_list]
    return filtered_boxes


import fitz

def pdf_to_image(pdf_path, output_path):
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image_list = page.get_pixmap()
        
        image_path = output_path.format(page_num)
        image_list.save(image_path)
    
    pdf_document.close()

def convertBase64FromPath(image_path):
    current_directory = os.getcwd()
    file_path = f"{current_directory}/{image_path}"
    # 打开图片文件
    with Image.open(file_path).convert("RGB") as img:
        # 将图片转换为字节流
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # 将字节流编码为base64
        img_base64 = base64.b64encode(img_byte_arr)
        img_base64_str = img_base64.decode('utf-8')  # 将字节类型转换为字符串
    return img_base64_str