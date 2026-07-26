from pathlib import Path

import cv2

import numpy as np

from template_Matching import imageSizes, detectPlayer, findBestObjectCoverage, loadTemplates, readGroundTruthBoxes


TemplateDir = Path("data/odevData/train/player")

TestPosDir = Path("data/odevData/test/positive")

TestNegDir = Path("data/odevData/test/negative")

TestLabelDir = Path("data/odevData/test/labels")


BestMatchThreshold = 0.50

BestBaseStepSize = 32

BestAreaThreshold = 0.05


# test görüntüsünü okumak için bu fonk
def readTestImage(imagePath):
    
    image = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {imagePath}")
    
    if image.shape != (imageSizes, imageSizes):
        image = cv2.resize(image, (imageSizes, imageSizes), interpolation=cv2.INTER_AREA)

    return image




# pozitif test görüntüsünü değerlendirmek için bu doğruysa 1 değilse 0 
def evaluatePosTestImages(imagePath, templates):

    image = readTestImage(imagePath)

    detection = detectPlayer(image, templates, BestMatchThreshold, BestBaseStepSize)

    if detection is None:
        return 0
    

    labelPath = (TestLabelDir / f"{imagePath.stem}.txt")

    groundTruthBoxes = (readGroundTruthBoxes(labelPath))

    predictedBox = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])

    bestCoverage, _ = (findBestObjectCoverage(predictedBox, groundTruthBoxes))

    if bestCoverage >= BestAreaThreshold:
        return 1
    
    return 0



# negatif test görüntüsünü değerlendirmek için burası
def evaluateNegTestImage(imagePath, templates):

    image = readTestImage(imagePath)

    detection = detectPlayer(image, templates, BestMatchThreshold, BestBaseStepSize)

    if detection is None:
        return 0 
    
    return 1



# metrikleri hesaplamak için bu fonksiyon
def calculateMetrics(TP, TN, FP, FN):

    total = (TP + TN + FP + FN)

    if total > 0: 
        accuracy = (TP + TN) / total

        mse = (FP + FN) / total
    else:
        accuracy = 0.0
        mse = 0.0

    precisionDenominator = (TP + FP)

    if precisionDenominator > 0:
        precision = (TP / precisionDenominator)
    else:
        precision = 0.0

    recallDenominator = (TP + FN)

    if recallDenominator > 0:
        recall = (TP / recallDenominator)
    else:
        recall = 0.0

    f1Denominator = precision + recall

    if f1Denominator > 0:
        f1Score = (2 * precision * recall / f1Denominator)
    else:
        f1Score = 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1Score": f1Score, "mse": mse}



# karmaşıklık matrisi için burası
def printConfusionMatrix(TP, TN, FP, FN):

    print("\nKarmaşıklık matrisi")

    print("                 Tahmin")
    print("              Oyuncu   Oyuncu Yok")
    print(f"Gerçek Oyuncu       {TP:4d}         {FN:4d}")
    print(f"Gerçek oyuncu yok {FP:4d}         {TN:4d}")




# bütün test verisi setini değerlendirmek için burası 
def evaluateTestDataset():

    templates = loadTemplates(TemplateDir)
    
    positivePaths = sorted(TestPosDir.glob("*.jpg"))

    negativePaths = sorted(TestNegDir.glob("*.jpg"))

    print(f"Template sayısı: {len(templates)}")

    print(f"Pozitif test görüntüsü: {len(positivePaths)}")

    print(f"Negatif test görüntüsü: {len(negativePaths)}")

    print(f"En iyi match threshold: {BestMatchThreshold}")

    print(f"En iyi step size: {BestBaseStepSize}")

    print(f"En iyi area threshold: {BestAreaThreshold}")

    trueP = 0
    trueN = 0
    falseP = 0
    falseN = 0

    for index, imagePath in enumerate(positivePaths, start=1):
        prediction = (evaluatePosTestImages(imagePath, templates))

        if prediction == 1:
            trueP += 1
        else:
            falseN += 1

        if index % 20 == 0:
            print(f"Pozitif test görüntüleri işlendi: {index}/{len(positivePaths)}")

    
    for index, imagePath in enumerate(negativePaths, start=1):
        prediction = (evaluateNegTestImage(imagePath, templates))

        if prediction == 0:
            trueN += 1
        else:
            falseP += 1

        if index % 50 == 0:
            print(f"Negatif test görüntüleri işlendi: {index}/{len(negativePaths)}")
        
    
    metrics = calculateMetrics(trueP, trueN, falseP, falseN)

    print("\n Test sonuçları")

    print(f"TP: {trueP}")
    print(f"TN: {trueN}")
    print(f"FP: {falseP}")
    print(f"FN: {falseN}")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"f1Score: {metrics['f1Score']:.4f}")
    print(f"Mse: {metrics['mse']:.4f}")

    printConfusionMatrix(trueP, trueN, falseP, falseN)


def main():
    evaluateTestDataset()

if __name__ == "__main__":
    main()