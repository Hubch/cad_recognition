import os
import labelMap

# 数据集路径
dataset_path = "input/labels"
label_file_extension = ".txt"

# 类别名称及其对应的编号
class_names = {v: k for k, v in labelMap.label_to_id.items()}


# 统计数据集中出现的类别
def count_classes(dataset_path, label_file_extension):
    class_count = {class_name: 0 for class_name in class_names.values()}
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(label_file_extension):
                label_file_path = os.path.join(root, file)
                with open(label_file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        class_id = int(line.split()[0])
                        class_name = class_names.get(class_id)
                        if class_name is not None:
                            class_count[class_name] += 1
    return class_count

# 比较配置的类别和数据集中统计的类别，找到缺失的类别
def find_missing_classes(config_classes, actual_classes):
    missing_classes = []
    for class_name in config_classes.values():
        if class_name not in actual_classes:
            missing_classes.append(class_name)
    return missing_classes

# 主函数
def main():
    # 统计数据集中出现的类别
    actual_class_count = count_classes(dataset_path, label_file_extension)
    
    # 比较配置的类别和数据集中统计的类别，找到缺失的类别
    missing_classes = find_missing_classes(class_names, actual_class_count)
    
    if missing_classes:
        print("缺失的类别：")
        for class_name in missing_classes:
            print(class_name)
    else:
        print("数据集中包含所有{}种配置的类别。".format(len(actual_class_count)))

if __name__ == "__main__":
    main()