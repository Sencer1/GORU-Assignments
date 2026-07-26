# BİL468 - ÖDEV3

Bu projede futbol görüntülerindeki nesneleri tespit etmek için sift öznitelikleri ve K-ağaç yöntemi kullanılmıştır. Görüntülerden çıkarılan  sift descriptorları, yolo formatındaki bounding box etiketlerine göre sınıflandırılmış ve bu descriptorlar kullanılarak K-ağaç modeli eğitilmiştir

K-ağaç yapısı hazır bir model kullanılmadan geliştirilmiştir. Ağacın her düğümündeki kümleme işlemi için yalnızca scikit_learn kütüphanesideki KMeans algoritması kullanılmıştır.

Kullanılan kütüphaneler:
-NumPy
-OpenCV
-Scikit-Learn
-Matplotlib
-Pickle
-Pathlib
-Pandas

Veri Seti
Projede yolo formatında etiketlenmiş dört nesne sınıfına sahip futbol görüntüleri kullanılmıştır.
Veri Seti train validation ve test olarak ayrlmıştır. Trainde 10000, validation ve testte 1500 görüntü bulunmaktadır.

Yolo etiket dosyalarındki her satır aşağıdaki biçimdedir:
class_id, center_x, center_y, width, height

Yolo sınıfları 0,1,2 ve 3 değerlerini kullanmaktadır. Modelde 0 değeri background sınıfına ayrıldığı için nesne sınıflarına bir eklenmiştir.

Veri Ön İşleme
Her görüntüde aşağıdaki işlemler gerçekleştirilmiştir:
1- Görüntü gri tonlamalı olarak okunmuştur.
2- Görüntü 128x128 poyutuna getirilmiştir.
3- Yolo etiketleri piksel kordinatlarına dönüştürülmüştür.
4- Görüntüden sift keypoint ve 128 boyutlu descriptorlar çıkarılmıştır.
5- Her sift descriptorı, keypoint merkezinin bulunduğu bounding box sınıfıyla etiketlenmiştir.
6- Hiçbir bounding box içerisinde bulunmayan descriptorlara background etiketi verilmiştir.
7- Birden fazla bounding box içerisinde bulunan descriptorlar eğitim verisinden çıkarılmıştır.

Sınıf Dengeleme
İlk özellik çıkarma işleminde sınıflar arasında büyük bir descriptor sayısı farkı oluşmuştur. Background ve sık görülen nesne sınıflarının modeli tamamen yönlendirmesini engellemek için her sınıftan 2000 descriptor kullanılmıştır. Yeterli descriptor bulunmayan sınıflarda tekrar örnekleme yapılmıştır.
Dengeleme sonrasında kullanılan descriptor dağılımı:
Background: 2000
Class 0: 2000
Class 1: 2000
Class 2: 2000
Class 3: 2000

Toplam 10.000 adet 128 boyutlu sift descriptorı kullanılmıştır.


K-Ağaç Modeli
K-ağaç sift descriptorlarını hiyerarşik olarak gruplandırmaktadır. 
Modelin iki temel hiperparametresi vardır:
-K : Her düğümde oluşturlan küme sayısı
-D : Ağacın maksimum derinliği

Proejde aşağıdaki değerler denenmiştir:
K_VALUES = [2, 4, 6]
DEPTH_VALUES = [2, 3, 4]

Eğitim sırasında bütün descriptorlar kök düğümde KMeans kullanılarak K kümeye ayrılır. Oluşan her küme için bir alt düğüm oluşturulur ve aynı işlem ağacın derinliği D değerine ulaşana kadar devam eder. Her düğümde, o düğüme ulaşan eğitim descriptorlarının sınıf sayıları tutulur. Tahmin sırasında descriptor kökten başlayarak KMeans merkezlerini takip eder ve bir yaprağa ulaşır. Yaprakta tutulan sınıf sayıları normalize edilerek sınıf olasılıklarına dönüştürülür.

Siliding Window
Validation ve test görüntüleri aşağıdaki pencere boyutlarıyla taranmıştır:
WINDOW_SIZES = [128, 64, 32, 16]

Her pencere boyutunda yüzde 50 örtüşme kullanılmıştır. Her pencere için:
1- Pencerinin s,ft descriptorları çıkarılır.
2- Descriptolar K-ağaç üzerinde ilgili yapraklara gönderilir.
3- Yapraklardaki sınıf olasılıları toplanır.
4- En yüksek olasılığa sahip sınıf pencere tahmini olarak seçilir.
5- Pencerenin gerçek sınıfı, bounding box ile arasındaki örtüşme oranına göre belirlenir.
6- Gerçek ve tahmin edilen sınıflar değerlendirme için kaydedilir.

Bir nesnenin en az %50'si pencerinin içerisinde bulunuyorsa pencere ilgili nesne sınıfına atanır. Aksi durumda bakcground olarak kabul edilir.


Değerlendirme 
Model performansı aşağıdaki metriklerle değerlendirilmiştir:
-Accuracy
-Precision
-Recall
-F1 Score
-Macro Precision
-Macro Recall
-Macro F1 Score
-Confusion Matrix
-Support

En iyi model seçilirken temel metrik olarak Macro F1 Score kullanılmıştır.

Her K ve D değeri için:
1- K-Ağaç modeli eğitilir.
2- Eğitim süresi kaydedilir.
3- Validation veri seti üzerinde tahmin yapılır.
4- Değerlendirme metrikleri hesaplanır.
5- En yüksek Macro F1 değerine sahip model kaydedilir.


PCA Görselleştirme
En iyi K-Ağaç modelindeki yaprak merkezleri ile rastgele seçilen eğitim descriptorları PCA kullanılarak iki boyuta indirgenmiştir. Grafikte eğitim descriptorları sınıflarına göre, K-Ağaç yaprak merkezleri ise büyük kare işaretlerle gösterilmektedir. Bu görsel, K-Ağaç bölgelerinin özellik uzayını ne kadar iyi ayırdığını incelemek amacıyla oluşturulmuştur.


Proje Dosyaları

src içindeki dosyalar:
config.py : Veri seti yollarını, çıktı dosyalarını, görüntü boyutunu, sınıf bilgilerini, K ve D değerlerini ve diğer proje ayarlarını içerir.

dataset.py : Görüntülerin okunması, yeniden boyutlandırılması, yolo etiketlerinin piksel kordinatlarına çevrilmesi ve pencere gerçek sınıflarının belirlenmesi işlemlerini gerçekleştirir.

sift_features.py : Görüntülerden sift descriptor çıkarır, descriptorları bounding box sınıflarıyla etiketler, sınıfları dengeler ve özellikleri npz dosyasına kaydeder.

k_tree.py : K-Ağaç düğüm ve model sınıflarını içerir. Ağacın eğitilmesi, descriptorların yapraklara göndeilmesi, sınıf olasılıklarının hesaplanması ve modelin kaydedilmesi işlemlerini gerçekleştirir.

detection.py : Sliding window bölgelerini oluşturur, validation ve test görüntüleri üzerinde tahmin yapar ve örnek görüntüde nesne tespiti gerçekleştirir.

metrics.py : Confusion matrix, accuracy, precision, recall, f1 ve macro metriklerini hesaplar. Confusion matrix görselini oluşturur.

train_k_tree.py : Sift eğitim özelliklerini hazırlar, farklı K ve D değerleriyle modelleri eğitir, validation sonuçlarını hesaplar, en iyi modeli seçer ve sonuç grafiklerini oluşturur.

test_evaluation.py : En iyi modeli test veri setinde değerlendirir, test metriklerini kaydeder ve seçilen bir test görüntüsü üzerinde tahmin kutularını çizer.

notebooks içindeki dosya:
results.ipynb : En iyi hiperparametrelerle elde edilen validation ve test sonuçlarını, confusion matrix görsellerini, PCA grafiğini ve iyi ve kötü örnek tespit sonucunu gösterir.


Oluşturulan çıktılar:
scale_results
bad_detection_example.jpg
best_k_tree.pkl
good_detection_example.jpg
pca_visualization.png
test_confusion_matrix.png
test_results_allscales.txt
train_sift_features.npz
validaiton_results.csv
validation_results.png


Çalıştırma sırası
Model eğitimi ve hiperparametre optimizasyonu için: python .\src\train_k_tree.py

En iyi modeli test veri setinde değerlendirmek için: python .\src\test_evaluation.py

