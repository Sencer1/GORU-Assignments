from pathlib import Path
import cv2
import numpy as np

from config import DETECTION_CONFIDENCE_THRESHOLD, IMAGE_SIZE, IOU_THRESHOLD, WINDOW_OVERLAP_RATIO, WINDOW_SIZES

from dataset import calculate_iou, find_image_files, find_window_true_class, read_and_preprocess_image, read_yolo_boxes

from k_tree import KTree

from sift_features import extract_region_descriptors



# tek bir pencere boyutu için silidng window bölgeleri üretmek için burası
def generate_windows_for_size(image, window_size):

    stride = int(window_size * (1.0 - WINDOW_OVERLAP_RATIO))

    stride = max(1, stride)

    for y1 in range(0, IMAGE_SIZE - window_size + 1 , stride):
        for x1 in range(0, IMAGE_SIZE - window_size + 1, stride):
            x2 = x1 + window_size
            y2 = y1 + window_size

            image_region = image[y1:y2, x1:x2]

            yield((x1, y1, x2, y2), image_region)




#  görüntüde 128 64 32 16 için %50 örtüşmeli sliding window üretmek için burası
def generate_sliding_windows(image):
    
    for window_size in WINDOW_SIZES:
        for window_box, image_region in generate_windows_for_size(image, window_size):
            yield(window_size, window_box, image_region)


# veri setini yalnızca belirtilen boyutunda değerlendirmek için burası
def evaluate_dataset_for_scale(model, image_directory, label_directory, window_size):

    true_labels = []
    predicted_labels = [] 

    image_paths = find_image_files(image_directory)

    for image_index, image_path in enumerate(image_paths, start=1):
        label_path = label_directory / f"{image_path.stem}.txt"

        image = read_and_preprocess_image(image_path)

        boxes = read_yolo_boxes(label_path)

        for window_box, image_region in generate_windows_for_size(image, window_size):

            descriptors = extract_region_descriptors(image_region)

            predicted_class, _ = model.predict_region(descriptors)

            true_class = find_window_true_class(window_box, boxes, IOU_THRESHOLD)

            true_labels.append(true_class)

            predicted_labels.append(predicted_class)

        if image_index % 50 == 0:
            print(f"{image_index}/{len(image_paths)} görüntü {window_size}x{window_size} ölçeğinde değerlendirildi.")

    return (np.asarray(true_labels, dtype=np.int32), np.asarray(predicted_labels, dtype=np.int32))


# bütün pencere boyutlarının sonuçlarını birleştirerek değerlendirmek için burası
def evaluate_dataset_all_scales(model, image_directory, label_directory):

    all_true_labels = []
    all_predicted_labels = []

    for window_size in WINDOW_SIZES:
        print(f"\n {window_size}x{window_size} ölçeği değerlendiriliyor.")

        true_labels, predicted_labels = evaluate_dataset_for_scale(model, image_directory, label_directory, window_size)

        all_true_labels.append(true_labels)

        all_predicted_labels.append(predicted_labels)

    return (np.concatenate(all_true_labels), np.concatenate(all_predicted_labels))


def evaluate_dataset(model, image_directory, label_directory):

    true_labels = []

    predicted_labels = []

    image_paths = find_image_files(image_directory)

    for image_index, image_path in enumerate(image_paths, start=1):

        label_path = (label_directory / f"{image_path.stem}.txt")

        image = read_and_preprocess_image(image_path)

        boxes = read_yolo_boxes(label_path)

        for _, window_box, image_region in generate_sliding_windows(image):
            descriptors = extract_region_descriptors(image_region)

            predicted_class, _ = model.predict_region(descriptors)

            true_class = find_window_true_class(window_box, boxes, IOU_THRESHOLD)

            true_labels.append(true_class)
            predicted_labels.append(predicted_class)

        if image_index % 50 == 0:
            print(f"{image_index}/{len(image_paths)} görüntü değerlendirildi.")

    return (np.asarray(true_labels,dtype=np.int32), np.asarray(predicted_labels, dtype=np.int32))




# aynı nesne üzerinde oluşan birden çok kutuyu azaltmak için burası
def non_maximum_suppression(detections, iou_threshold=0.30):

    if not detections:
        return[]

    detections = sorted(detections, key=lambda detection: detection["confidence"], reverse=True)

    selected_detections = []

    while detections:
        best_detection = detections.pop(0)

        selected_detections.append(best_detection)

        remaining_detections = []

        for detection in detections:

            if detection["class_id"] != best_detection["class_id"]:
                remaining_detections.append(detection)
                continue

            iou = calculate_iou(best_detection["box"], detection["box"])

            if iou < iou_threshold:
                remaining_detections.append(detection)

        detections = remaining_detections

    return selected_detections 


# görüntüdeki nesne tahminlerini bulmak için burası
def detect_objects(model, image):

    detections = []

    for window_size, window_box, image_region in generate_sliding_windows(image):

        descriptors = extract_region_descriptors(image_region)

        predicted_class, confidence = model.predict_region(descriptors)

        if predicted_class == 0:
            continue

        if confidence < DETECTION_CONFIDENCE_THRESHOLD:
            continue

        detections.append(
            {
                "box": window_box,
                "class_id": predicted_class,
                "confidence": confidence,
                "window_size": window_size
            }
        )

    return non_maximum_suppression(detections, iou_threshold=0.30)


# tahmin kutularının gerçek kutularla eşleşme kalitesini hesaplamak için burası
def calculate_detection_quality(detections, boxes):

    if not detections:
        return 0.0

    total_quality = 0.0

    for detection in detections:
        detection_box = detection["box"]
        detection_class = detection["class_id"]

        best_iou = 0.0

        for box in boxes:
            if box.class_id != detection_class:
                continue

            object_box = (box.x1, box.y1, box.x2, box.y2)

            current_iou = calculate_iou(detection_box, object_box)

            if current_iou > best_iou:
                best_iou = current_iou

        total_quality = total_quality + best_iou

    return total_quality / len(detections)
