from pathlib import Path
import cv2
import numpy as np

from config import SIFT_MAX_FEATURES
from dataset import find_image_files, find_point_class, read_and_preprocess_image, read_yolo_boxes


#  sifti oluşturmka için burası
def create_sift():
    return cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    # sıft objesi oluşturuyor nfeature da en fazla kaç tane keypoint tutulacağını belirlemek için



# burası bir görüntüden sift descriptor çıkartmak ve onu etiketlemek için
def extract_labeled_descriptors(image, boxes):

    sift = create_sift()

    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None or len(keypoints) == 0:
        return (np.empty((0,128), dtype=np.float32), np.empty((0,), dtype=np.int32))
    

    selected_descriptors = []
    selected_labels = []

    for keypoint, descriptor in zip(keypoints, descriptors):
        x, y = keypoint.pt

        class_id = find_point_class(x, y, boxes)

        if class_id is None:
            continue

        selected_descriptors.append(descriptor)
        selected_labels.append(class_id)

    if not selected_descriptors:
        return (np.empty((0,128), dtype=np.float32), np.empty((0,), dtype=np.int32))
    
    return (np.asarray(selected_descriptors, dtype=np.float32), np.asarray(selected_labels, dtype=np.int32))



# kaydırılan pencereden sift descriptor çıkarmak için burası
def extract_region_descriptors(image_region):

    sift = create_sift()


    _, descriptors  = sift.detectAndCompute(image_region, None)

    if descriptors is None:
        return np.empty((0,128), dtype=np.float32)

    return descriptors.astype(np.float32)




# background descriptor sayısını obje descriptor sayısına göre azalatmak için burası
def balance_feature_dataset(features, labels):

    random_generator = np.random.default_rng(50)

    unique_classes, class_counts = np.unique(labels, return_counts=True)

    print("\n Sınıf sayıları:")

    for class_id, class_count in zip(unique_classes, class_counts):

        print(f"Sınıf {class_id}: {class_count}")

    target_count = 2000

    balanced_feature_parts = []
    balanced_label_parts = []

    for class_id in unique_classes:
        class_indices = np.where(labels == class_id)[0]

        replace = len(class_indices) < target_count

        selected_indices = random_generator.choice(class_indices, size=target_count, replace=replace)

        balanced_feature_parts.append(features[selected_indices])
        balanced_label_parts.append(labels[selected_indices])

    balanced_features = np.concatenate(balanced_feature_parts, axis=0)

    balanced_labels = np.concatenate(balanced_label_parts, axis=0)

    shuffle_indices = random_generator.permutation(len(balanced_features))

    balanced_features = balanced_features[shuffle_indices]

    balanced_labels = balanced_labels[shuffle_indices]

    return balanced_features, balanced_labels






# bbütün görüntülerin sift özelliklerini çıkarmak için burası
def build_feature_dataset(image_directory, label_directory, output_path):

    image_paths = find_image_files(image_directory)

    all_descriptors = []
    all_labels = []


    for image_index, image_path in enumerate(image_paths, start=1):
        label_path = label_directory / f"{image_path.stem}.txt"

        image= read_and_preprocess_image(image_path)
        boxes = read_yolo_boxes(label_path)


        descriptors, labels = extract_labeled_descriptors(image, boxes)

        if len(descriptors) > 0:
            all_descriptors.append(descriptors)
            all_labels.append(labels)


        if image_index % 100 == 0:
            print(f"{image_index}/{len(image_paths)} görüntü işlendi.")


    if not all_descriptors:
        raise ValueError(f"Hiç sift özelliği çıkarılamadı: {image_directory}")


    features = np.concatenate(all_descriptors, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("\n Dengeleme öncesi sınıf sayıları:")

    unique_classes, class_counts = np.unique(labels, return_counts=True)

    for class_id, class_count in zip(unique_classes, class_counts):
        print(f"Sınıf {class_id}: {class_count}")

    features, labels = balance_feature_dataset(features, labels)

    print("\n Dengeleme sonrası sınıf sayıları:")

    unique_classes, class_counts = np.unique(labels, return_counts=True)

    for class_id, class_count in zip(unique_classes, class_counts):
        print(f"Sınıf {class_id}: {class_count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


    np.savez_compressed(output_path, features=features, labels=labels)

    print(f"\n Özellikler dosyası kaydedildi: {output_path}")
    print(f"Toplam descriptors sayısı: {len(features)}")
    print(f"Descriptor boyutu: {features.shape[1]}")



# burası kaydedilmiş sift özelliklerini yüklemek için
def load_feature_dataset(feature_path):

    data = np.load(feature_path)

    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int32)

    return features,labels


