# -*- coding: utf-8 -*-

import os
import json

json_dir = 'output'
txt_dir = 'input/labels'

label_to_id = {'仪表': 0,
 '装置': 1,
 '界区连续': 2,
 '法兰': 3,
 '法兰盖': 4,
 '焊帽': 5,
 '管帽': 6,
 '软管': 7,
 '8字盲板': 8,
 '同心异径管': 9,
 '偏心异径管': 10,
 '敞开漏斗': 11,
 '封闭漏斗': 12,
 '放空管': 13,
 '带放空帽放空': 14,
 '无弯曲联接': 15,
 '可曲扰接头': 16,
 '扰性软管': 17,
 '传力接头': 18,
 '快接头': 19,
 '静态混合器': 20,
 '混合三通': 21,
 '喷嘴': 22,
 '穿孔管': 23,
 '过滤器': 24,
 'Y型过滤器': 25,
 'T型过滤器': 26,
 '闸阀': 27,
 '球阀': 28,
 '止回阀': 29,
 '蝶阀': 30,
 '电动闸阀': 31,
 '气动闸阀': 32,
 '电动蝶阀': 33,
 '气动蝶阀': 34,
 '水力控制阀': 35,
 '调节阀': 36,
 '截止阀': 37,
 '隔膜阀': 38,
 '安全阀': 39,
 '钢制伸缩器': 40,
 '滤网': 41,
 '喇叭口': 42,
 '角座阀': 43,
 '刀型阀': 44,
 '电磁阀': 45,
 '气动两位三通阀': 46,
 '疏水阀': 47,
 '排气阀': 48,
 '水表': 49,
 'PFD泵': 50,
 '介质单向流向': 51,
 '离心泵': 52,
 '立式容器': 53,
 '往复泵': 54,
 '背压阀': 55,
 '螺杆泵': 56,
 '消音器': 57,
 '管道混合器': 57,
 '真空射流器': 59}

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_json_file(json_file, txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    with open(txt_file, 'w', encoding='utf-8') as f:
        for shape in data['shapes']:
            label = shape['label']
            if label in label_to_id:
                class_id = label_to_id[label]
                points = shape['points']
                box = [points[0][0], points[0][1], points[1][0], points[1][1]]
                yolo_box = convert_to_yolo((image_width, image_height), box)
                f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

def convert_all_json_to_yolo(json_dir, txt_dir):
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    for json_filename in os.listdir(json_dir):
        if json_filename.endswith('.json'):
            json_file = os.path.join(json_dir, json_filename)
            txt_filename = os.path.splitext(json_filename)[0] + '.txt'
            txt_file = os.path.join(txt_dir, txt_filename)
            process_json_file(json_file, txt_file)

# 执行转换
convert_all_json_to_yolo(json_dir, txt_dir)
