from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import CLASS_COUNT, CLASS_NAMES


# gerçek ve tahmin edilen sınıflardan karmaşıklık matrisi oluşturmak için
def calculate_confusion_matrix(true_labels, predicted_labels):

    confusion_matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)

    for true_label, predicted_label in zip(true_labels, predicted_labels):

        confusion_matrix[int(true_label), int(predicted_label)] += 1


    return confusion_matrix


# accurcy precision recall ve f1 hesabı için  
def calculate_metrics(confusion_matrix):


    total_samples = confusion_matrix.sum()
    correct_predictions = np.trace(confusion_matrix)
    # trace matrix köşegendeki elemanları toplar

    if total_samples == 0:
        accuracy = 0.0
    else:
        accuracy = correct_predictions / total_samples

    class_precisions = []
    class_recalls = []
    class_f1_scores = []


    for class_id in range(CLASS_COUNT):
        true_positive = confusion_matrix[class_id, class_id]

        false_positive = (confusion_matrix[:, class_id].sum() - true_positive) 

        false_negative = (confusion_matrix[class_id, :].sum() - true_positive)

        precision_denominator = (true_positive + false_positive)

        recall_denominator =(true_positive + false_negative)

        if precision_denominator == 0:
            precision = 0.0
        else:
            precision = (true_positive / precision_denominator)

        if recall_denominator == 0:
            recall = 0
        else:
            recall = (true_positive / recall_denominator)


        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = (2* precision * recall / (precision + recall))

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1_scores.append(f1_score)


    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(class_precisions)),
        "macro_recall": float(np.mean(class_recalls)),
        "macro_f1": float(np.mean(class_f1_scores)),
        "class_precisions": class_precisions,
        "class_recalls": class_recalls,
        "class_f1_scores": class_f1_scores
    }



# confusion matrisini ve sınıf metrikleini ekrana basmak için burası
def print_classification_results(confusion_matrix, metrics):

    print("\n Confusion Matrix")
    print(confusion_matrix)

    print("\n Genel sonuçlar")
    print("-" * 50)
    print(f"Accuracy: {metrics["accuracy"]:.4f}")
    print(f"Macro Precision: {metrics["macro_precision"]:.4f}")
    print(f"Macro Recall: {metrics["macro_recall"]:.4f}")
    print(f"Macro F1: {metrics["macro_f1"]:.4f}")


    print("\n Sınıf Sonuçları")
    print("-" * 50)

    for class_id in range(CLASS_COUNT):
        class_name = CLASS_NAMES[class_id]

        precision = metrics["class_precisions"][class_id]

        recall = metrics["class_recalls"][class_id]

        f1_score = metrics["class_f1_scores"][class_id]

        support = confusion_matrix[class_id, :].sum()

        print(f"{class_name:<15}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1_score:.4f}")
        print(f"Support: {support}")




# confusion matrixi görsel olarak kaydetmek için burası
def save_confusion_matrix_graph(confusion_matrix, output_path, title):

    figure, axis = plt.subplots(figsize=(8, 7))

    matrix_image = axis.imshow(confusion_matrix, interpolation="nearest")

    figure.colorbar(matrix_image, ax=axis)

    class_name = [CLASS_NAMES[class_id] for class_id in range(CLASS_COUNT)]

    axis.set(xticks=np.arange(CLASS_COUNT), yticks=np.arange(CLASS_COUNT), xticklabels=class_name, yticklabels=class_name, xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf", title=title)
    #  tek seferde birden fazla özelliği ayarlamak için axis için

    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    # bir veya daha fazla matplotlib nesnesinin özelliklerini ayarlamak için

    threshold = confusion_matrix.max() / 2

    for row_index in range(CLASS_COUNT):
        for column_index in range(CLASS_COUNT):
            value = confusion_matrix[row_index, column_index]

            if value > threshold:
                text_color = "white"
            else:
                text_color = "black"

            axis.text(column_index, row_index, str(value), ha="center", va="center", color=text_color)

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(figure)





