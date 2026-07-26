from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_SIZE

# nesnenin sınıfını ve sınırlarını tutması için
# datacalss ile __init__ yazmamıza gerek kalmıyor
@dataclass
class BoundingBox:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


# görüntü klasöründe görüntüleri bulmak için
def find_image_files(image_directory):

    supported_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []

    for path in image_directory.iterdir():
        if path.suffix.lower() in supported_extensions:
            # suffix uzantıyı verir .jpg gibi lower küçük harf yapar
            image_paths.append(path)

    return sorted(image_paths)


# görüntüyü gri yapar ve 128x128 yapar burası
def read_and_preprocess_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


    return image



# yolo etiketlerini 128x128 görüntü üzerinde piksel kordinatlarına çevirmke için burası
def read_yolo_boxes(label_path):

    boxes = []

    if not label_path.exists():
        return boxes
    
    with label_path.open("r", encoding="utf-8") as label_file:
        for line in label_file:
            values = line.strip().split()
            # baş ve sondaki boşukları silmek için strip
            #  split boşluk boşluk ayırmak için

            if len(values) != 5:
                continue

            yolo_class_id = int(values[0])
            center_x = float(values[1])
            center_y = float(values[2])
            width = float(values[3])
            height = float(values[4])

            center_x = center_x * IMAGE_SIZE
            center_y = center_y * IMAGE_SIZE
            width = width * IMAGE_SIZE
            height = height * IMAGE_SIZE

            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            x1 = max(0, min(x1, IMAGE_SIZE - 1))
            y1 = max(0, min(y1, IMAGE_SIZE - 1))
            x2 = max(0, min(x2, IMAGE_SIZE))
            y2 = max(0, min(y2, IMAGE_SIZE))

            if x2 <= x1 or y2 <= y1:
                continue

            
            model_class_id = yolo_class_id + 1

            boxes.append(BoundingBox(model_class_id, x1, y1, x2, y2))


    return boxes


# burası sıft noktasının hangi bounding boxda olduğunu belirlemek için
# hiçbir kutuda değilse 0 döndürüyor yada  bir kutudaysa nesnenin sınıfını döndürür
def find_point_class(x, y, boxes):
    containing_classes = []

    for box in boxes:
        if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2:
            containing_classes.append(box.class_id)

    if len(containing_classes) == 0:
        return 0
    
    if len(containing_classes) == 1:
        return containing_classes[0]
    
    return None


# burası iki kutu arasındaki iou hesabı için
def calculate_iou(first_box, second_box):
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)


    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    first_area = (max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1))
    second_area = (max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1))

    union_area = first_area + second_area - intersection_area

    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


# iou kullanarak sınıf belirlemek için burası eşiğin altındaysa background oluyor 
# window kaydırarak uyguluyoruz
def find_window_true_class(window_box, boxes, iou_threshold):
    
    window_x1, window_y1, window_x2, window_y2 = window_box

    best_overlap_ratio = 0.0
    best_class_id = 0

    for box in boxes:
        intersection_x1 = max(window_x1, box.x1)
        intersection_y1 = max(window_y1, box.y1)
        intersection_x2 = min(window_x2, box.x2)
        intersection_y2 = min(window_y2, box.y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2- intersection_y1)

        intersection_area = intersection_width * intersection_height

        object_area = max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)

        if object_area == 0:
            continue

        overlap_ratio = intersection_area / object_area

        if overlap_ratio > best_overlap_ratio:
            best_overlap_ratio = overlap_ratio
            best_class_id = box.class_id

    if best_overlap_ratio < iou_threshold:
        return 0

    return best_class_id