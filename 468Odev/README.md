Bil468 ÖDEV 1

Proje Konusu
Bu çalışmada futbol maç görüntülerinde oyuncu tespiti için çok ölçekli şablon eşleştirme yöntemi uygulanmıştır. Çalışmada yalnızca player sınıfı kullanılmıştır.

Veri Seti Yapısı

data/original
Orijinal SoccerNet veri setini içeriyor.
train: Orjinal train görüntüleri ve etiketleri
valid: Orjinal validation görüntüleri ve etiketleri
test: Orijinal test görüntüleri ve etiketlerini
data.yaml: Sınıf isimlerini ve veri seti klasörlerini içeren yapılandırma dosyası

Veri setindeki sınıflar:
0: player
1: goalkeeper
2: refree
3: ball

Bu ödevde yalnızca player sınıfı kullanıldı.

data/odevData
Ödev için orijinal veri setinden oluşturulan geçici veri setini içerir

odevDta/train
Oyuncuların görüntünün büyük bölümünü kaplayacak şekilde kırpıldığı 10 adet eğitim şablonunu içerir.

odevData/validation
positive: Oyuncu içeren 200 validation görüntüsü
negative: Oyuncu içermeyen 600 validation görüntüsü
labels: Pozitif validation görüntülerinin güncellenmiş konum etiketleri

odevData/test
positive: Oyuncu içeren 200 test görüntüsü
negative: Oyuncu içermeyen 600 test görüntüsü
labels: Pozitif test görüntülerinin güncellenmiş konum etiketleri

Bütün ödev görüntüleri gri tonlamalı ve 256x256 boyutundadır.

Proje Dosyaları

prepare_Dataset.py
Orijinal SoccerNet görüntülerinden ödev veri setini oluşturur.

Bu dosya:
Yolo etiketlerini okur.
Yalnızca player sınıfına ait kutuları seçer.
Normalize Yolo kordinatlarını piksel kordinatlarına dönüştürür.
Train için 10 oyuncu şablonu oluşturur.
Validation için 299 pozitif ve 600 negatif görüntü üretir.
Test için 299 pozitif ve 600 negatif görüntü üretir.
Pozitif görüntülerin konum etiketlerini kırpılan görüntülere göre günceller.
Görüntüleri gri tonlamalı ve 256x256 boyutunda kaydeder.

template_Matching.py
Çok ölçekli şablon eşleştirme yöntemini uygular.

Bu dosya:
10  oyuncu şablonunu yükler
Şablonları 256x256, 128x128, 64x64, 32x32 ölçeklerinde kullanır.
Şablon ölçeğine göre kayma miktarını hesaplar.
cv2.matchTemplate ve TM_CCOEFF_NORMED yöntemini kullanır.
En yüksek benzerlik skoruna sahip bölgeyi seçer.
Tahmin kutusunun gerçek oyuncu kutusunu kapsama oranını hesaplar.
Seçilen test görüntüsüde oyuncu sınıfını ve tahmin kutusunu gösterir.

validation_Experiments.py
Validation veri setinde hiperparametre deneylerini gerçekleştirir.
Deneylerde değiştirilen hiperparametreler:
Eşleşme eşiği 
Temel kayma miktarı
Alan yüzdesi eşiği

Her deney için şu değerler hesaplanır:
TP, TN, FP, FN, Accuracy, Precision, Recall, F1 skor, Mse

test_Evaluation.py
Validation deneylerinde seçilen en iyi hyperparametreler kullanarak test veri setini değerlendirir.

Test değerlendirmesinde:
TP, TN, FP, FN, Accuracy, Precision, Recall, F1 skor, Mse, Karmaşıklık matrisi hesaplanır.

homework_Results.ipynb
Ödev sonuçlarının çalıştırılması ve kaydedilmiş olarak gösterildiği Jupyter Notebook dosyasıdır.

Notebook içinde bulunanlar:
Veri seti sayılar
10 eğitim şablonu
Validation deney sonuçları
En iyi hyperparametrelenin seçimi
Test sonucu
Karmaşıklık matrisi
Örnek test tahmini
Sonuç değerlendirmesi

playerBoxTest.jpg
Orijinal görüntüde player sınıfına ait Yolo etiketlerinin doğru okunup okunmadığını kontrol etmek amacıyla oluşturulan görüntüdür.

finalTestPrediction.jpg
Seçilen test görüntüsündeki şablon eşleştirme sonucunu gösterir.
Kırmızı kutu: Tahmin edilen oyuncu bölgesi
Yeşil kutu: Gerçek oyuncu etiketi
player yazısı: Tahmin edilen nesne sınıfı

En iyi hyperparametreler
Validation deneylerinde en yüksek f1 skorunu veren değerler:
Eşleşme eşiği: 0.50
Temel kayma miktarı: 32
Alan yüzdesi eşiği: 0.05

Test Sonuçları
TP: 74
TN: 48
FP: 552
FN 126
Accuracy: 0.1525
Precision: 0.1182
Recall: 0.3700
F1 skor: 0.1792 
Mse: 0.8475

Çalıştırma Sırası

Ödev veri setini yeniden oluşturmak için:
python prepare_Dataset.py

Seçilen bir görüntü üzerinde şablon eşleştirme yapmak için:
python template_Matching.py

Validation deneylerini çalıştırmak için:
python validation_Experiments.py

En iyi hyperparametreler ile test veri setini değerlendirmek için:
python test_Evaluation.py

Sonuçları görüntülemek için homework_Results.ipynb dosyasındaki hücreler sırayla çalıştırılmalı.