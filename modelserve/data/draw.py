import os
import cv2
import labelMap

def load_labels(label_path):
    """
    Load YOLO label file.
    """
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return [list(map(float, label.strip().split())) for label in labels]

def draw_bounding_box(image, bbox, class_name):
    """
    Draw bounding box on the image.
    """
    x_center, y_center, width, height = bbox
    img_h, img_w = image.shape[:2]

    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def visualize_yolo_dataset(image_dir, label_dir, output_dir, class_dict):
    """
    Visualize YOLO dataset and save images to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
        
        if not os.path.exists(label_path):
            continue
        
        image = cv2.imread(image_path)
        labels = load_labels(label_path)
        
        for label in labels:
            class_id, bbox = int(label[0]), label[1:]
            class_name = class_dict[class_id]
            draw_bounding_box(image, bbox, str(class_id))

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)

# Example usage
image_dir = 'input/images'
label_dir = 'input/labels'
output_dir = 'input/labeled_images'
class_dict = {v: k for k, v in labelMap.label_to_id.items()}

visualize_yolo_dataset(image_dir, label_dir, output_dir, class_dict)
