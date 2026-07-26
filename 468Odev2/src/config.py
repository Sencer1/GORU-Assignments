# ortak ayarları yazmak için burası

# burda yapılacaklar 
# klasör yolları, görüntü boyutu, pencere boyutu, kayma miktarı, iou eşiği, sınıf sayısı, model ve çıktıların yolu

from pathlib import Path
# klasör yollarını yazmak için burası

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# __file__ şu anki dosyayı temsil ediyor 
# .resolce() config.py nin tam yolunu alır
# parent ile üstünün üstüne çıkar
# projenin klasöronun yolunu aldık yani

DATA_ROOT = PROJECT_ROOT / "data" / "homework"

TRAIN_IMAGE_DIR = DATA_ROOT / "train" / "images"
TRAIN_LABEL_DIR = DATA_ROOT / "train" / "labels"

VAL_IMAGE_DIR = DATA_ROOT / "val" / "images"
VAL_LABEL_DIR = DATA_ROOT / "val" / "labels"

TEST_IMAGE_DIR = DATA_ROOT / "test" / "images"
TEST_LABEL_DIR = DATA_ROOT / "test" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

# görüntü ve pencere boyutu için burası
IMAGE_SIZE = 256
WINDOW_SIZE = 128
WINDOW_STEP = 64

# eşik değeri için burası
IOU_THRESHOLD = 0.20

# o pencerede oyuncu yoksa sınıfın numrası için
BACKGROUND_CLASS = 0

# gerçek nesne sınıfı için burası
NUM_OBJECT_CLASSES = 4
NUM_CLASSES = NUM_OBJECT_CLASSES + 1

RANDOM_SEED = 42

BEST_MODEL_PATH = MODEL_DIR / "best_mlp_model.pt"

CLASS_NAME = [
    "background",
    "player",
    "ball",
    "referee",
    "goalkeeper"
]


