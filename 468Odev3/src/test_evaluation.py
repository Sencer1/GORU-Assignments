import cv2 
from config import BAD_DETECTION_RESULT_PATH, BEST_MODEL_PATH, CLASS_NAMES,COMBINED_TEST_CONFUSION_MATRIX_PATH,COMBINED_TEST_RESULT_PATH,GOOD_DETECTION_RESULT_PATH,SCALE_RESULTS_DIRECTORY,TEST_IMAGES_DIRECTORY,TEST_LABELS_DIRECTORY,WINDOW_SIZES
from dataset import find_image_files, read_and_preprocess_image, read_yolo_boxes
from detection import calculate_detection_quality, detect_objects, evaluate_dataset_all_scales, evaluate_dataset_for_scale
from k_tree import KTree
from metrics import calculate_confusion_matrix, calculate_metrics, print_classification_results, save_confusion_matrix_graph


# metrikleri ve confusion matrixi metin dosyasına kaydetmek için burası
def save_results_text(confusion_matrix, metrics, output_path, title):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as result_file:
        result_file.write(f"{title}\n")

        result_file.write(f"=" * 50 + "\n\n")

        result_file.write(f"Accuracy: {metrics["accuracy"]:.4f}\n")

        result_file.write(f"Macro Precision: {metrics["macro_precision"]:.4f}\n")

        result_file.write(f"Macro Recall: {metrics["macro_recall"]:.4f}\n")

        result_file.write(f"Macro F1: {metrics["macro_f1"]:.4f}\n\n")

        result_file.write("Confusion Matrix\n")

        result_file.write(str(confusion_matrix))

        result_file.write("\n\n Sınıf Sonuçları \n")

        result_file.write("-" * 50 + "\n")

        for class_id in range(len(metrics["class_f1_scores"])):

            support = confusion_matrix[class_id, :].sum()

            result_file.write(f"{CLASS_NAMES[class_id]}\n")

            result_file.write(f"Precision: {metrics["class_precisions"][class_id]:.4f}\n")

            result_file.write(f"Recall: {metrics["class_recalls"][class_id]:.4f}\n")

            result_file.write(f"F1: {metrics["class_f1_scores"][class_id]:.4f}\n")

            result_file.write(f"Support: {support}\n\n")



# 128 64 32 16 pencere boyutlarını ayrı ayrı değerlendirmek için burası
def evaluate_each_scale(model):

    SCALE_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for window_size in WINDOW_SIZES:
        print("\n" + "=" * 50)

        print(f"Test ölçeği: {window_size}x{window_size}")

        print("=" * 50)

        true_labels, predicted_labels = evaluate_dataset_for_scale(model, TEST_IMAGES_DIRECTORY, TEST_LABELS_DIRECTORY, window_size)

        confusion_matrix = calculate_confusion_matrix(true_labels, predicted_labels)

        metrics = calculate_metrics(confusion_matrix)

        print_classification_results(confusion_matrix, metrics)

        result_text_path = SCALE_RESULTS_DIRECTORY / f"test_results_{window_size}.txt"

        confusion_image_path = SCALE_RESULTS_DIRECTORY / f"test_confusion_matrix_{window_size}.png"

        save_results_text(confusion_matrix, metrics, result_text_path, title=(f"Test Sonuçları: {window_size}x{window_size}"))

        save_confusion_matrix_graph(confusion_matrix, confusion_image_path, title=(f"Test Confusion Matrix - {window_size}x{window_size}"))





# bütün ölçekleri bir arada değerlendirmek için burası
def evaluate_combined_scales(model):

    print("\n" + "=" * 50)

    print("Bütün ölçeklerin birleşik test değerlendirmesi")

    print("=" * 50)

    true_labels, predicted_labels = evaluate_dataset_all_scales(model, TEST_IMAGES_DIRECTORY, TEST_LABELS_DIRECTORY)

    confusion_matrix = calculate_confusion_matrix(true_labels, predicted_labels)

    metrics = calculate_metrics(confusion_matrix)

    print_classification_results(confusion_matrix, metrics)

    save_results_text(confusion_matrix, metrics, COMBINED_TEST_RESULT_PATH, title=("Bütün ölçeklerin birleşik test sonuçları"))

    save_confusion_matrix_graph(confusion_matrix, COMBINED_TEST_CONFUSION_MATRIX_PATH, title=("Bütün ölçeklerin birleşik confusion matrixi"))




# tahmin kutularını görüntü üzerine çizmek için burası
def draw_detection_image(image, detections, output_path):

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]

        class_id = detection["class_id"]

        confidence = detection["confidence"]

        window_size = detection["window_size"]

        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = (f"{CLASS_NAMES[class_id]} {confidence:.2f} {window_size}x{window_size}")

        cv2.putText(color_image, label_text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), color_image)



# Test görüntüleri arasında en yüksek ve en düşük kalite puanına sahip örnekelri seçip kaydetmek için burası
def create_good_and_bad_examples(model):

    image_paths = find_image_files(TEST_IMAGES_DIRECTORY)

    best_quality = -1.0
    worst_quality = float("inf")

    best_image = None
    worst_image = None

    best_detections = None
    worst_detections = None

    best_image_name = None
    worst_image_name = None

    for image_index, image_path in enumerate(image_paths, start=1):

        label_path = TEST_LABELS_DIRECTORY / f"{image_path.stem}.txt"

        image = read_and_preprocess_image(image_path)

        boxes = read_yolo_boxes(label_path)

        detections = detect_objects(model, image)

        quality = calculate_detection_quality(detections, boxes)

        if quality > best_quality:
            best_quality = quality
            best_image = image.copy()
            best_detections = detections
            best_image_name = image_path.name


        if quality < worst_quality:
            worst_quality = quality
            worst_image = image.copy()
            worst_detections = detections
            worst_image_name = image_path.name

        if image_index % 100 == 0:

            print(f"{image_index}/{len(image_paths)} görüntü iyi-kötü örnek için incelendi.")


    if best_image is not None:

        draw_detection_image(best_image, best_detections, GOOD_DETECTION_RESULT_PATH)

    if worst_image is not None:
        draw_detection_image(worst_image, worst_detections, BAD_DETECTION_RESULT_PATH)


    print(f"\n İyi örnek: {best_image_name}")

    print(f"İyi örnek kalite puanı: {best_quality:.4f}")

    print(f"Kötü örnek: {worst_image_name}")

    print(f"Kötü örnek kalite puanı: {worst_quality:.4f}")



def main():

    model = KTree.load(BEST_MODEL_PATH)

    evaluate_each_scale(model)

    evaluate_combined_scales(model)

    create_good_and_bad_examples(model)

    print("\n Test değerlendirmesi tamamlandı.")

    print(f"Ölçek sonuçları: {SCALE_RESULTS_DIRECTORY}")

    print(f"Birleşik sonuç: {COMBINED_TEST_RESULT_PATH}")

    print(f"İyi örnek: {GOOD_DETECTION_RESULT_PATH}")

    print(f"Kötü örnek: {BAD_DETECTION_RESULT_PATH}")


if __name__ == "__main__":
    main()
