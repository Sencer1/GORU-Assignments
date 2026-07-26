from pathlib import Path

import cv2

import numpy as np
#  görüntü matrisleri ve sayısal işlemler için

imageSizes = 256

templateSizes = (256, 128, 64, 32)

# kaydırma miktarı
baseStepSize = 32

matchThreshold = 0.50
# min benzerlik skoru bu

templateDir = Path("data/odevData/train/player")

sampleImagePath = Path("data/odevData/test/positive/positive_0002.jpg")

outputPath = Path("finalTestPrediction.jpg")


sampleLabelPath = Path("data/odevData/test/labels/positive_0002.txt")

areaThreshold = 0.05

# templateleri  okumak için bu fonksiyon
def loadTemplates(templateDirectory):

    templatePaths = sorted(templateDirectory.glob("*.png"))

    templates = []

    for templatePath in templatePaths:

        template = cv2.imread(str(templatePath), cv2.IMREAD_GRAYSCALE)

        if template is None:
            print(f"Template okunamadı: {templatePath}")
            continue

        templates.append(template)
    
    if not templates:
        raise RuntimeError("Hiçbir template okunamadı")
    
    return templates


# template göre kayma miktarını ayarlamak için burası
def calculateStepSize(templateSize, BaseStepSize):

    scale = templateSize / imageSizes

    stepSize = int(BaseStepSize * scale)

    return max(1, stepSize)



# tek template ve ölçek üzerinden eşleştirme. en çok benzediği yeri ve benzerlik puanını döndürme fonksiyonu
def matchSingleTemplate(image, template, stepSize):

    # template in tüm görünte üzerinde gezmesi için bu matchTemplate
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # cv2.TM_CCOEFF_NORMED benzerlik hesaplama yöntemi sonuuç -1 ve 1 arasında

    # kaymalı olarak sonuç matrisiden değer almakm için kaçar kaçar kayacağını verdik
    sampleResult = result[::stepSize, ::stepSize]

    maxIndex = np.unravel_index(np.argmax(sampleResult), sampleResult.shape)
    # np.argmax matrisi düz tek sıraya çevirip en yüksek değerli indeski bulur 1,2,8 -- 3,4,5 ---> 2
    # np.unravel_index de düz satır olan indexleri satır sutüna çevirir ve nuamra verir 2, (2,3) ---> (0,2)

    sampleY, sampleX = maxIndex

    # stepli sonuç matrisden gerçek kordinatlara çevirmek için burası
    bestX = sampleX * stepSize
    bestY = sampleY * stepSize

    # en iyi skorun sol üst köşesi
    bestScore = float(sampleResult[sampleY, sampleX])

    return bestScore, (bestX, bestY)



# bütün template ve ölçekleri deneyerek en yüksek skorlu sonuç için bu fonksiyon
def detectPlayer(image, templates, MatchThreshold, BaseStepSize):

    bestDetection = None
    # hem elemanları hem de indexleri dönemk için
    for templateIndex, template in enumerate(templates, start=1):

        for templateSize in templateSizes:
            resizedTemplates = cv2.resize(template, (templateSize, templateSize), interpolation=cv2.INTER_AREA)
            # templateleri her size göre ayarlıyor

            templateHeight, templateWidth = (resizedTemplates.shape[:2])

            imageHeight, imageWidth = (image.shape[:2])

            if(templateWidth > imageWidth or templateHeight > imageHeight):
                continue

            stepSize = calculateStepSize(templateSize, BaseStepSize)

            score, location = (matchSingleTemplate(image, resizedTemplates, stepSize))

            if(bestDetection is None or score > bestDetection["score"]):
                x, y = location

                bestDetection = {"score": score, "templateIndex": (templateIndex), "templateSize": (templateSize), "stepSize": stepSize, "x1": x, "y1": y, "x2": x + templateWidth, "y2": y + templateHeight}

    if bestDetection is None:
        return None
    
    if(bestDetection["score"] < MatchThreshold):
        return None
    
    return bestDetection



# yolo etiketlerinden piksel değerşeri olarak kutular için
def readGroundTruthBoxes(labelPath):

    groundTruthBoxes = []

    if not labelPath.exists():
        return groundTruthBoxes
    
    with labelPath.open("r", encoding="utf-8") as labelFile:
        for line in labelFile:
            values = line.strip().split()

            if len(values) != 5:
                continue

            classId = int(values[0])

            if classId != 0:
                continue

            centerX = float(values[1]) * imageSizes
            centerY = float(values[2]) * imageSizes
            boxWidth = float(values[3]) * imageSizes
            boxHeight = float(values[4]) * imageSizes

            x1 = int(centerX - boxWidth / 2)
            y1 = int(centerY - boxHeight / 2)
            x2 = int(centerX + boxWidth / 2)
            y2 = int(centerY + boxHeight / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(imageSizes, x2)
            y2 = min(imageSizes, y2)

            groundTruthBoxes.append((x1, y1, x2, y2))

    return groundTruthBoxes



# iki kutunun kesişim alanını hespalamak için burası
def calculateIntersectionArea(firstBox, secondBox):

    fx1, fy1, fx2, fy2 = firstBox
    sx1, sy1, sx2, sy2 = secondBox

    intersectionX1 = max(fx1, sx1)
    intersectionY1 = max(fy1, sy1)
    intersectionX2 = min(fx2, sx2)
    intersectionY2 = min(fy2, sy2)

    intersectionWidth = max(0, intersectionX2 - intersectionX1)
    intersectionHeight = max(0, intersectionY2- intersectionY1)

    return intersectionWidth * intersectionHeight
    

# tahmin edilen kutunun gerçek örneğin gerçek kutus ile arasında ne kadarını kapsadığını hesapladığını ölçmek için
def calculateObjectCoverage(predictedBox, groundTruthBox):

    groundTruthX1, groundTruthY1, groundTruthX2, groundTruthY2 = groundTruthBox

    groundTruthWidth = (groundTruthX2 - groundTruthX1)
    groundTruthHeight = (groundTruthY2 - groundTruthY1)
    
    groundTruthArea = (groundTruthWidth * groundTruthHeight)

    if groundTruthArea <= 0:
        return 0.0
    
    intersectionArea = calculateIntersectionArea(predictedBox, groundTruthBox)

    coverage = (intersectionArea / groundTruthArea)

    return coverage



# birden fazla gerçek örneğin arasından en uygununu bulmak için burası
def findBestObjectCoverage(predictedBox, groundTruthBoxes):

    bestCoverage = 0.0
    bestGroundTruthBox = None

    for groundTruthBox in groundTruthBoxes:
        coverage = calculateObjectCoverage(predictedBox, groundTruthBox)

        if coverage > bestCoverage:
            bestCoverage = coverage
            bestGroundTruthBox = groundTruthBox
    
    return bestCoverage, bestGroundTruthBox
    






#  sonuçları görüntüye çevirmek için burası artık gerçek kutuyu da çizicez
def drawDetection(image, detection, groundTruthBox):
    
    outputImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]
    
    cv2.rectangle(outputImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # sol üst ve sağ alt köşeye çizmek için

    labelText = (f"player: {detection['score']:.3f}")

    textY = max(20, y1 - 10)

    # metin yazmak içn burası
    cv2.putText(outputImage, labelText, (x1, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)


    if groundTruthBox is not None:
        gx1, gy1, gx2, gy2 = groundTruthBox

        cv2.rectangle(outputImage, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

        cv2.putText(outputImage, "ground truth", (gx1, max(20, gy1- 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


    return outputImage


def main():

    templates = loadTemplates(templateDir)
    #  templateleri tek tek okuduk burda

    print(f"okunan template sayısı: {len(templates)}")

    image = cv2.imread(str(sampleImagePath), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {sampleImagePath}")

    if image.shape != (imageSizes, imageSizes):
        image = cv2.resize(image, (imageSizes, imageSizes), interpolation=cv2.INTER_AREA)

    
    groundTruthBoxes = (readGroundTruthBoxes(sampleLabelPath))

    print(f"Gerçek oyuncu kutu sayısı: {len(groundTruthBoxes)}")

    detection = detectPlayer(image, templates, matchThreshold, baseStepSize)



    if detection is None:
        print("Görüntüde template bulunamadı ")

        print("Görüntü gerçekte oyuncu içerdiği için sonuç FN")

        onay = cv2.imwrite(str(outputPath), image)
    else:
        print("template bulundu")

        print(f"Benzerlik skoru: {detection['score']:.4f}")

        print(f"Kullanılan template: {detection['templateIndex']}")

        print(f"template boyutu: {detection['templateSize']} x {detection['templateSize']}")

        print(f"Kayma miktarı: {detection['stepSize']}")

        print(f"Tahmin kordinatları: {detection['x1']}, {detection['y1']}, {detection['x2']}, {detection['y2']}")

        predictedBox = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])

        bestCoverage, bestGroundTruthBox = findBestObjectCoverage(predictedBox, groundTruthBoxes)

        print(f"Gerçek oyuncu kapsama oranı: {bestCoverage:.4f}")

        print(f"Gerçek oyuncu kapsama yüzdesi: %{bestCoverage * 100:.2f}")

        if bestCoverage >= areaThreshold:
            print("Konum tahmini doğru kabul edildi TP")
        else:
            print("Konum tahmini yanlış kabul edildi FN")



        outputImage = drawDetection(image, detection, bestGroundTruthBox)

        onay = cv2.imwrite(str(outputPath), outputImage)

    if not onay:
        raise RuntimeError("Sonuç görüntüsü kaydedilemedi")
    
    print(f"Sonuç kaydedildi: {outputPath}")


if __name__ == "__main__":
    main()
