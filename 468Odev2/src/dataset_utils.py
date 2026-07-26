# burdaki amaçlar şunlar

# yolo label dosyalrını okuma
# yolo dan piksel kordinat dönüşümü
# 256x256 görüntüden 9 pencere oluşturmak için
# her pencereye obejct ya da background etiketi atamak için


from pathlib import Path

import cv2

import numpy as np

from config import IMAGE_SIZE, WINDOW_SIZE, WINDOW_STEP, IOU_THRESHOLD, BACKGROUND_CLASS


# burda resme göre kordinatlarını aldık
def yoloToPixelCor(centerX, centerY, width, height, imageWidth, imageHeight):

    boxWidth = width * imageWidth
    boxHeight = height * imageHeight

    x1 = int((centerX * imageWidth) - (boxWidth / 2))
    y1 = int((centerY * imageHeight) - (boxHeight / 2))
    x2 = int((centerX * imageWidth) + (boxWidth / 2))
    y2 = int((centerY * imageHeight) + (boxHeight / 2))

    x1 = max(0, min(imageWidth - 1 , x1))
    y1 = max(0, min(imageHeight - 1 , y1))
    x2 = max(0, min(imageWidth - 1 , x2))
    y2 = max(0, min(imageHeight - 1 , y2))

    return x1, y1, x2, y2

# yolo dan label dosyalarını okumak için
def readYoloLabels(labelPath):

    boxes = []

    if not labelPath.exists():
        return boxes
    
    with labelPath.open("r", encoding="utf-8") as file:
        lines = file.readlines()
        # tüm satırları okumak için burası

    for line in lines:
        parts = line.strip().split()
        # strip baş ve sondaki boşlukları siler split listeye dönüştürmek için

        if len(parts) != 5:
            continue


        classId = int(float(parts[0]))
        centerX = float(parts[1])
        centerY = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x1, y1, x2, y2 = yoloToPixelCor(centerX,centerY,width,height,IMAGE_SIZE,IMAGE_SIZE)

        boxes.append((classId, x1, y1, x2, y2))

    return boxes



# görüntüyü okuyup boyut ve renk değiştirmek için burası
def readAndPreprocessImage(imagePath):
    
    image = cv2.imread(str(imagePath))

    if image is None:
        raise ValueError(f"Görüntü okunamadı: {imagePath}")
    

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resizedImage = cv2.resize(grayImage, (IMAGE_SIZE, IMAGE_SIZE))

    return resizedImage

# 256x256 resim üzerinde 9 pencere üretmek için burası
def createWindows():
    windows = []

    for y1 in range(0, IMAGE_SIZE - WINDOW_SIZE + 1, WINDOW_STEP):
        # 0 129 64 şeklinde 3 pencere
        for x1 in range(0, IMAGE_SIZE - WINDOW_SIZE + 1, WINDOW_STEP):
            x2 = x1 + WINDOW_SIZE
            y2 = y1 + WINDOW_SIZE

            windows.append((x1, y1, x2, y2))

    return windows


# iki kutu arasındaki iou değerini hesaplamak için burası
def calculateIou(boxA, boxB):

    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    intersectionX1 = max(ax1, bx1)
    intersectionY1 = max(ay1, by1)
    intersectionX2 = max(ax2, bx2)
    intersectionY2 = max(ay2, by2)

    intersectionWidth = max(0, intersectionX2 - intersectionX1)
    intersectionHeight = max(0, intersectionY2 - intersectionY1)

    intersectionArea = intersectionWidth * intersectionHeight

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    unionArea = areaA + areaB - intersectionArea

    if unionArea == 0:
        return 0.0
    
    return intersectionArea / unionArea


# pencereye etiket atamak için
# pencerede nesne yoksa 0 eğer iou eeşiğiyle bir nesne varsa 1 döner
# birden fazla nesne varsa none olarak çıkarır

def assignWindowLabel(window, boxes, removeMultiObject=True):
    matchedClasses = []

    for classId, x1, y1, x2, y2 in boxes:
        objectBox = (x1, y1, x2, y2)
        iou = calculateIou(window, objectBox)

        if iou >= IOU_THRESHOLD:
            matchedClasses.append(classId + 1)

    if len(matchedClasses) == 0:
        return BACKGROUND_CLASS
    
    if len(matchedClasses) == 1:
        return matchedClasses[0]
    
    if removeMultiObject:
        return None
    
    return matchedClasses[0]


# görüntü klasöründeki dosayalrı bulmak için
def findImageFiles(imageDir):
    imagePaths = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        imagePaths.extend(imageDir.glob(extension))
    # extend hepsini eklemek için glob da sonu öyle biten dosyalrı bulmak için 
    # append tek tek ekliyordu extend bulduklarının hespini tekte ekler
    # liste içine liste ekler yani append extend ise o lsiteyi tek tek s
    return sorted(imagePaths)