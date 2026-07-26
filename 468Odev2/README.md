
#  BIL 468 2. ÖDEV

HOG öznitelikleri ve mlp ile nesne tespiti

Projenin amacı, görünte bölgelerinden çıkarılan hog öznitelikleri kullanarak pytroch ile eğitilen bir mlp modeliyle nesne tespiti yapmak.

Görüntüler önce gri tonlmaya çevirlir ve 256x256 boyutuna yeniden boyutlandırılır. Daha sonra görüntü 128x128 boyutunda 9 pencereye ayrılır. Pencereler 64 piksel kayma miktarıyla oluşur. Her pencere için hog özniteliklleri çıkarılır ve mlp ye input olarak verilir.

Model her pencere için orda nesne var mı diye kontrol eder.

Veri seti data klasöründe tutulmaktadır
5600 train 1200 val ve 1200 test görüntüsünden oluşmaktadır

her görüntünün bir de yolo label dosyası bulunmaktadır.

Dosya açıklamaları

config.py
Projedeki ortak ayarları içerir. Veri seti yolları, görüntü boyutu, pencere boyutu, kayma miktarı, ıou eşiği, sınıf sayısı ve çıktı yolları bu dosyada tanımlandı.

dataset_utils.py
Veri seti işlemleri için yardımcı fonksiyonları içerir. Görüntü okuma, gri tona çevirme, yeniden boyutlandırma, yolo etiket okuma ve çevirme, 9 pencere oluşturma ıou hesaplama ve pencereye etiket atama işlemleri burada yapılır.

hog_features.py
Hog öznitelik çıkarma işlmelerini içerir. Her 128x128 ikiye bölünür ve her parçadan hog öznitelikleri çıkarılır daha sonra bu iki parça birleştirilir 7560 boyutlu öznitelik çıkarılır.

build_features.py
Train, val ve test veri setleri için hog özniteliklleri çıkarılır burda. Her görüntü 9 pencereye ayrılır her pencereye ıou ile etiket verilir ve npz dosyalarına kaydedilir.

mlp_model.py
Pytorch ile oluşturulan mlp modelini içeriri burası. Model 7560 hog öznitelik vektörünü alır ve pencerenin hangi class olduğunu tahmin eder.

train_mlp.py
Mlp modelini eğitir daha sonra train val özelliklerini yükler bu özellikleri standardize eder. Sınıf ağırlıklarını hesaplar, farklı hyperparametreleri dener ve early stopping uygular. En sonda da en iyi modeli best mlp model olarak kaydeder.

test_evalution.py
Eğitilen en iyi modeli test veri testi ile değerlendirir. Accuracy, precision, recall, f1, karmaşıklık matrisi oluşturup sonuçları output result a kaydeder.

visualize_detection.py
Seçilen bir test görüntüsü üzerinden modeli çalıştırır. Görüntüyü 9 pencereye ayrırır ve her pencere için tahmin yapılır nesne bulunan ögeler kare içine alınır. Çıktı görseli output result a kaydedilir.

Çalıştırma sırası şu şekilde

python src/build_features.py -> python src/train_mlp.py -> python src/test_evaluation.py -> python src/visualize_detection.py


Çıktılar da şu şekilde oluşur
outputs/features/train_features.npz
outputs/features/val_features.npz
outputs/features/test_features.npz
outputs/models/best_mlp_model.pt
outputs/results/test_metrics.txt
outputs/results/detection_*.jpg