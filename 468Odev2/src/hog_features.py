# burası hog feature çıkarmak için
# pencere boyutu 128x128
# bir pencere ikiye bölnür 64x128
# ayrımdan 3780 lik hog çıkarılır
# iki bölgenin hog u birleştirilir sonuç olarak 7560 feature vektörü elde edilir

import cv2
import numpy as np

from config import WINDOW_SIZE

# hog görüntüdeki kenarların çizgilerin yönlerin dağılımını çıkarmak için

# hog un nasıl çalışacağını belirleyen ayarlar için burası
def createHogDescriptor():
    
    hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    # winsize gelecek görüntü boyutu burdan feature çıkarılacak
    # cellsize hog görüntüyü küçük hücrelere böler
    # nbins her cell içinde kenar yönlerinin kaç gruba ayrılacağını belirler 9 farklı bilgi tutuyoruz yani
    # block size block içinde birden fazla cell in birleşmiş hali
    # block her adımda kaç piksel kayacak ona bakıyor

    return hog

def extractWindowHogFeatures(image, window, hog):

    x1, y1, x2, y2 = window

    windowImage = image[y1:y2, x1:x2]

    if windowImage.shape != (WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError(f"Pencere boyutu hatalı: {windowImage.shape}")
    

    leftRegion = windowImage[:, 0:64]
    rightRegion = windowImage[:, 64:128]

    leftFeatures = hog.compute(leftRegion).flatten()
    rightFeatures = hog.compute(rightRegion).flatten()
    # flatten düz sayı sırası olarak almak için

    windowFeature = np.concatenate([leftFeatures, rightFeatures])
    # sol ve sağ featureları birleştirdik
    return windowFeature.astype(np.float32)