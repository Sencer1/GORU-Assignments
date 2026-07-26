import csv 
import time
import matplotlib.pyplot as plt
import numpy as np

from config import BEST_MODEL_PATH, DEPTH_VALUES, K_VALUES, MODEL_DIRECTORY, OUTPUT_DIRECTORY, PCA_GRAPH_PATH, PCA_SAMPLE_COUNT, RANDOM_SEED, TRAIN_FEATURE_PATH, TRAIN_IMAGES_DIRECTORY, TRAIN_LABELS_DIRECTORY, VALIDATION_FEATURE_PATH, VALIDATION_GRAPH_PATH, VALIDATION_IMAGES_DIRECTORY, VALIDATION_LABELS_DIRECTORY, VALIDATION_RESULTS_PATH

from detection import evaluate_dataset
from k_tree import KTree
from metrics import calculate_confusion_matrix, calculate_metrics, print_classification_results
from sift_features import build_feature_dataset, load_feature_dataset


# eğitim ve validation için sift dosylarını oluşturmka için burası
def prepare_feature_files():

    if not TRAIN_FEATURE_PATH.exists():
        print("\n Eğitim sift özellikleri çıkartılıyor.")

        build_feature_dataset(TRAIN_IMAGES_DIRECTORY, TRAIN_LABELS_DIRECTORY, TRAIN_FEATURE_PATH)



# hiperparametreleri csv dosyasına katdetmek için burası
def save_validation_results(results):


    with VALIDATION_RESULTS_PATH.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=["k_value", "depth", "training_time", "validation_accuracy", "validation_macro_f1"])
        # dictleri direkt csv ye yazmaaya yarıyor dictwriter

        writer.writeheader()
        # sütun başlıklaırnı en başa yazmak için
        writer.writerows(results)
        # sonuçları döngüsüz tek seferde yazmak için


# k ve d değerlerine göre validation f1 ve eğitim süresini göstermek için burası
def create_validation_graph(results):

    experiment_names = [f"K={result["k_value"]}, D={result["depth"]}" for result in results]

    validation_f1_scores = [result["validation_macro_f1"] for result in results]

    training_times = [result["training_time"] for result in results]

    x_positions = np.arange(len(results))

    figure, first_axis = plt.subplots(figsize=(12, 6))

    first_axis.plot(x_positions, validation_f1_scores, marker="o", label="Validation Macro F1")

    first_axis.set_xlabel("K VE D Değerleri")

    first_axis.set_ylabel("Validation Macro F1")

    first_axis.set_xticks(x_positions)

    first_axis.set_xticklabels(experiment_names, rotation=45, ha="right")

    second_axis = first_axis.twinx()

    second_axis.plot(x_positions, training_times, marker="s", linestyle="--", label="Eğitim Süresi")

    second_axis.set_ylabel("Eğitim Süresi (saniye cinsinden)")

    first_axis.set_title("K ve D değerlerine göre validaiton performansı")

    figure.tight_layout()

    figure.savefig(VALIDATION_GRAPH_PATH, dpi=200, bbox_inches="tight")

    plt.close(figure)



# pca yi numpy ile yapmak için burası
def calculate_pca_projection(data, component_count=2):

    mean_vector = np.mean(data, axis=0)

    centered_data = data - mean_vector

    _, _, right_singular_vectors = np.linalg.svd(centered_data, full_matrices=False)

    principal_comnponents = right_singular_vectors[:component_count]

    projected_data = centered_data @ principal_comnponents.T

    return projected_data


# yaprak merkezlerini ve rastgele descriptorları pca ile iki boyutta göstermek için burası
def create_pca_graph(model, training_features, training_labels):

    leaf_centers = model.collect_leaf_centers()

    if len(leaf_centers) == 0:
        print("PCA grafiği içn yaprak merkezi bulunamadı.")
        return


    random_generator = np.random.default_rng(RANDOM_SEED)

    sample_count = min(PCA_SAMPLE_COUNT, len(training_features))

    sample_indices = random_generator.choice(len(training_features), sample_count, replace=False)

    sample_features = training_features[sample_indices]

    sample_labels = training_labels[sample_indices]

    combined_data = np.concatenate([sample_features, leaf_centers], axis=0)

    projected_data = calculate_pca_projection(combined_data)

    projectec_samples = projected_data[:sample_count]

    projected_centers = projected_data[sample_count:]

    figure, axis = plt.subplots(figsize=(10, 8))

    unique_classes = np.unique(sample_labels)

    for class_id in unique_classes:

        class_mask = sample_labels == class_id

        axis.scatter(projectec_samples[class_mask, 0], projectec_samples[class_mask, 1], s=15, alpha=0.60, label=f"Sınıf{class_id}")

    axis.scatter(projected_centers[:, 0], projected_centers[:, 1], marker="s", s=120, edgecolors="black", linewidths=1.5, label="K-Agaç Yaprak Markezleri")

    axis.set_title("SIFT özellikleri ve K-Ağaç yaprak merkezleri")

    axis.set_xlabel("Birinci PCA Bileşeni")

    axis.set_ylabel("İkinci PCA Bileşeni")

    axis.legend()

    figure.tight_layout()

    figure.savefig(PCA_GRAPH_PATH, dpi=200, bbox_inches="tight")

    plt.close(figure)



def main():

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    prepare_feature_files()

    train_features, train_labels = load_feature_dataset(TRAIN_FEATURE_PATH)

    print(f"\n Eğitim descriptor sayısı: {len(train_features)}")
    print(f"Descriptor boyutu: {train_features.shape[1]}")

    results = []

    best_model = None
    best_macro_f1 = -1.0
    best_k_value = None
    best_depth = None 

    for k_value in K_VALUES:
        for depth in DEPTH_VALUES:

            print("\n" + "=" * 50)

            print(f"K-Ağaç eğitiliyor: K={k_value}, D={depth}")

            print("=" * 50)

            model = KTree(k_value, depth)

            training_start_time = time.perf_counter()

            model.fit(train_features, train_labels)

            training_time = time.perf_counter() - training_start_time

            print(f"Eğitim süresi: {training_time:.2f} saniye")

            print("\n Validation değerlendirmesi yapılıyor.")

            true_labels, predicted_labels = evaluate_dataset(model, VALIDATION_IMAGES_DIRECTORY, VALIDATION_LABELS_DIRECTORY)

            confusion_matrix = calculate_confusion_matrix(true_labels, predicted_labels)

            metrics = calculate_metrics(confusion_matrix)

            print_classification_results(confusion_matrix, metrics)

            current_result = {"k_value": k_value, "depth": depth, "training_time": round(training_time, 4), "validation_accuracy": round(metrics["accuracy"], 6), "validation_macro_f1": round(metrics["macro_f1"], 6)}

            results.append(current_result)

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]

                best_k_value = k_value
                best_depth = depth
                best_model = model

    if best_model is None:
        raise RuntimeError("En iyi model seçilemedi.")

    best_model.save(BEST_MODEL_PATH)

    save_validation_results(results)

    create_validation_graph(results)

    create_pca_graph(best_model, train_features, train_labels)

    print("\n" + "=" * 50)

    print("En iyi model")

    print("-" * 50)

    print(f"K değeri: {best_k_value}")
    print(f"D değeri: {best_depth}")

    print(f"Validation Macro F1: {best_macro_f1:.4f}")

    print(f"Model kaydedildi: {BEST_MODEL_PATH}")

    print(f"Validation grafiği: {VALIDATION_GRAPH_PATH}")

    print(f"PCA grafiği: {PCA_GRAPH_PATH}")


if __name__ == "__main__":
    main()






