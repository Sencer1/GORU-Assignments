from pathlib import Path

import cv2

import numpy as np

from template_Matching import imageSizes, detectPlayer, findBestObjectCoverage, loadTemplates, readGroundTruthBoxes


TemplateDir = Path("data/odevData/train/player")

ValidationPosDir = Path("data/odevData/validation/positive")

ValidationNegDir = Path("data/odevData/validation/negative")

ValidationLabelDir = Path("data/odevData/validation/labels")

MatchThreshold = 0.20

BaseStepSize = 2

AreaThreshold = 0.10


# validationdaki görüntüyü okumak için bu fonksiyon
def readValidationImage(imagePath):
    image = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {imagePath}")
    
    if image.shape != (imageSizes, imageSizes):
        image = cv2.resize(image, (imageSizes, imageSizes), interpolation=cv2.INTER_AREA)

    return image



#  tek pozitif görüntüyü değerlendirmek için burası doğruysa 1 yoksa 0 
def evaluatePositiveImage(imagePath, templates):

    image = readValidationImage(imagePath)

    detection = detectPlayer(image, templates, MatchThreshold, BaseStepSize)

    if detection is None:
        return 0
    
    labelPath = (ValidationLabelDir / f"{imagePath.stem}.txt")

    groundTruthBoxes = (readGroundTruthBoxes(labelPath))

    predictedBox = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])

    bestCoverage, _ = (findBestObjectCoverage(predictedBox, groundTruthBoxes))

    if bestCoverage >= AreaThreshold:
        return 1

    return 0


# tek negatif görüntüyü değerlendirmek için burası da 
def evaluateNegativeImage(imagePath, templates):

    image = readValidationImage(imagePath)

    detection = detectPlayer(image, templates, MatchThreshold, BaseStepSize)

    if detection is None:
        return 0
    
    return 1



# burası da metrikleri hesaplamak için 
def calculateMetrics(TP, TN, FP, FN):

    total = (TP + TN + FP + FN)

    if total > 0:
        accuracy = ((TP + TN) / total)
    else: 
        accuracy = 0.0

    precisionDenominator = (TP + FP)

    if precisionDenominator > 0:
        precision = (TP / precisionDenominator)
    else: 
        precision = 0.0

    recallDominator = (TP + FN)

    if recallDominator > 0:
        recall = (TP / recallDominator)
    else:
        recall = 0.0

    f1Dominator = precision + recall

    if f1Dominator > 0:
        f1Score = (2 * precision * recall / f1Dominator)
    else: 
        f1Score = 0.0


    if total > 0:
        mse = (FP + FN) / total
    else: 
        mse = 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1Score": f1Score, "mse": mse}


# bütün validation kümesini değerlendirmek için burası
def evaluateValidationDataset():
    templates = loadTemplates(TemplateDir)

    positivePaths = sorted(ValidationPosDir.glob("*.jpg"))

    negativePaths = sorted(ValidationNegDir.glob("*.jpg"))

    print(f"Temaplate sayısı: {len(templates)}")

    print(f"Pozitif görüntü sayısı: {len(positivePaths)}")

    print(f"Negatif görüntü sayısı: {len(negativePaths)}")

    trueP = 0
    falseN = 0
    trueN = 0
    falseP = 0

    for index, imagePath in enumerate(positivePaths, start=1):
        prediction = evaluatePositiveImage(imagePath, templates)

        if prediction == 1:
            trueP += 1
        else:
            falseN += 1

        if index % 20 == 0:
            print(f"Pozitif görüntüler işlendi: {index}/{len(positivePaths)}")
        

    for index, imagePath in enumerate(negativePaths, start=1):
        prediction = evaluateNegativeImage(imagePath, templates)

        if prediction == 0:
            trueN += 1
        else:
            falseP += 1
        
        if index % 50 == 0:
            print(f"Negatif görüntüler işlendi: {index}/{len(negativePaths)}")

    
    metrics = calculateMetrics(trueP, trueN, falseP, falseN)

    print("\n Validation Sonuçları")

    print(f"Benzerlik eşiği: {MatchThreshold}")

    print(f"Temel kayma miktarı: {BaseStepSize}")

    print(f"Alan eşiği: {AreaThreshold}")

    print(f"TP: {trueP}")
    print(f"TN: {trueN}")
    print(f"FP: {falseP}")
    print(f"FN: {falseN}")
    

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"f1 score: {metrics['f1Score']:.4f}")
    print(f"Mse: {metrics['mse']:.4f}")


def main():
    evaluateValidationDataset()

if __name__ == "__main__":
    main()