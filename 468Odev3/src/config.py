from pathlib import Path


# bu projenin ana klasörü
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# __file__ çalışan pyhton dosyasının yolunu verir
# resolve tam dosya yolunu verir c:\...... uzun hali
# paarents da iki üst yolunu verir


# data klasörleri için burası
DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_IMAGES_DIRECTORY = DATA_ROOT / "train" / "images"
TRAIN_LABELS_DIRECTORY = DATA_ROOT / "train" / "labels"

VALIDATION_IMAGES_DIRECTORY = DATA_ROOT / "validation" / "images"
VALIDATION_LABELS_DIRECTORY = DATA_ROOT / "validation" / "labels"

TEST_IMAGES_DIRECTORY = DATA_ROOT / "test" / "images"
TEST_LABELS_DIRECTORY = DATA_ROOT / "test" / "labels"

# model ve çıktı klasörleri için burası
MODEL_DIRECTORY = PROJECT_ROOT / "models"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"


# sıft featurların kaydedileciği dosyalar  
TRAIN_FEATURE_PATH = OUTPUT_DIRECTORY / "train_sift_features.npz"
VALIDATION_FEATURE_PATH = OUTPUT_DIRECTORY / "validation_sift_features.npz"


# en iyi modeli kaydetmek için bu dosya

BEST_MODEL_PATH = OUTPUT_DIRECTORY / "best_k_tree.pkl"

# hiperparametre sonuçlarını tutmak için bunlar da
VALIDATION_RESULTS_PATH = OUTPUT_DIRECTORY / "validation_results.csv"
VALIDATION_GRAPH_PATH = OUTPUT_DIRECTORY / "validation_results.png"

# pca görseli için burası da
PCA_GRAPH_PATH = OUTPUT_DIRECTORY / "pca_visualization.png"


# test sonuçları için burası da
TEST_CONFUSION_MATRIX_PATH = OUTPUT_DIRECTORY / "test_confusion_matrix.png"
TEST_RESULT_PATH = OUTPUT_DIRECTORY / "test_result.txt"
DETECTION_RESULT_PATH = OUTPUT_DIRECTORY / "detection_result.jpg"

SCALE_RESULTS_DIRECTORY = OUTPUT_DIRECTORY / "scale_results"

COMBINED_TEST_CONFUSION_MATRIX_PATH = OUTPUT_DIRECTORY / "test_confusion_matrix_all_scale.png"

COMBINED_TEST_RESULT_PATH = OUTPUT_DIRECTORY / "test_results_all_scales.txt"

GOOD_DETECTION_RESULT_PATH = OUTPUT_DIRECTORY / "good_detection_example.jpg"

BAD_DETECTION_RESULT_PATH = OUTPUT_DIRECTORY / "bad_detection_example.jpg"


# görüntü boyutu
IMAGE_SIZE = 128

# window boyutları
WINDOW_SIZES = [128, 64, 32, 16]

# windor için örtüşme oranı
WINDOW_OVERLAP_RATIO = 0.50

# sıft feature uzunluğu
SIFT_FEATURE_SIZE = 128

# sınıf bilgileri burası

BACKGROUND_CLASS_ID = 0
OBJECT_CLASS_COUNT = 4
CLASS_COUNT = OBJECT_CLASS_COUNT + 1

IOU_THRESHOLD = 0.50

# her çalıştırmada sonuçların aynı çıkması için burası  
RANDOM_SEED = 50

# görüntü başına çıkacak en fazla sıft özelliği
SIFT_MAX_FEATURES = 80

CLASS_NAMES = {
    0: "Background",
    1: "Class 0",
    2: "Class 1",
    3: "Class 2",
    4: "Class 3"
}

# k ve derinlik değerleri burası 
K_VALUES = [2, 4, 6]
DEPTH_VALUES = [2, 3, 4]

# burası kmeans ayarları
KMEANS_MAX_ITERATIONS = 100
KMEANS_N_INIT = 5

# testte nesne kabulu için gereken olasılık burası
DETECTION_CONFIDENCE_THRESHOLD = 0.50

# pca grafiğinde gösterilecek descriptor sayısı için burası
PCA_SAMPLE_COUNT = 1000



