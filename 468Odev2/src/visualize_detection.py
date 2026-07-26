# seçilen test görüntüsü üzerinde modeli çalıştırmka için

import cv2
import numpy as np

import torch

from config import TEST_IMAGE_DIR, BEST_MODEL_PATH, RESULT_DIR,IMAGE_SIZE, CLASS_NAME
from dataset_utils import readAndPreprocessImage, createWindows,findImageFiles

from hog_features import createHogDescriptor, extractWindowHogFeatures
from mlp_model import HOGMLP



def calculateIou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    interX1 = max(ax1, bx1)
    interY1 = max(ay1, by1)
    interX2 = max(ax2, bx2)
    interY2 = max(ay2, by2)

    interWidth = max(0, interX2 - interX1)
    interHeight = max(0, interY2- interY1)

    interArea = interWidth * interHeight

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    unionArea = areaA + areaB - interArea

    if unionArea == 0:
        return 0.0
    
    return interArea / unionArea


def applyNms(detections, iouThreshold=0.25):

    detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)

    selectedDetections = []

    for detection in detections:
        keep = True

        for selected in selectedDetections:
            sameClass = detection["className"] == selected["className"]
            overlap = calculateIou(detection["box"], selected["box"])

            if sameClass and overlap > iouThreshold:
                keep = False
                break

        if keep:
            selectedDetections.append(detection)

    return selectedDetections


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imagePaths = findImageFiles(TEST_IMAGE_DIR)
    if len(imagePaths) == 0:
        raise ValueError("Test klasörünce görüntü bulunamadı")
    
    # burda test görüntüsü seçiyoruz
    imagePath = imagePaths[160]

    grayImage = readAndPreprocessImage(imagePath)
    colorImage = cv2.imread(str(imagePath))
    colorImage = cv2.resize(colorImage, (IMAGE_SIZE, IMAGE_SIZE))

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)


    scaler = checkpoint["scaler"]


    model = HOGMLP(
        inputSize= checkpoint["input_size"],
        numClasses= checkpoint["num_classes"],
        hiddenSize= checkpoint["best_params"]["hiddenSize"],
        hiddenLayers= checkpoint["best_params"]["hiddenLayers"],
        useBatchNorm= checkpoint["best_params"]["useBatchNorm"],
        dropoutRate= checkpoint["best_params"]["dropoutRate"]
    ).to(device)


    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hog = createHogDescriptor()
    windows = createWindows()

    detections = []

    for window in windows:
        features = extractWindowHogFeatures(grayImage, window, hog)

        featuresScaled = scaler.transform(features.reshape(1, -1))
        # tek örnek 7560 -> 1 , 7560 a çevirmek için rehsape kısmı

        featuresTensor = torch.tensor(featuresScaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(featuresTensor)

            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

            confidence = confidence.item()
            prediction = prediction.item()

            # prediction = torch.argmax(output, dim=1).item()

            if prediction != 0 and confidence >= 0.90:
                x1, y1, x2, y2 = window

                className = CLASS_NAME[prediction]

                detections.append(
                    {
                        "box" : (x1, y1, x2, y2),
                        "className" : className,
                        "confidence" : confidence
                    }
                )
        
        detections = applyNms(detections, iouThreshold=0.25)

        detections = detections[:4]

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            className = detection["className"]
            confidence = detection["confidence"]

            cv2.rectangle(colorImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(colorImage, f"{className} {confidence:.2f}", (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                # cv2.rectangle(colorImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # className = CLASS_NAME[prediction]
                # cv2.putText(colorImage, f"{className}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    outputPath = RESULT_DIR / f"detection_{imagePath.stem}.jpg"
    cv2.imwrite(str(outputPath), colorImage)

    print(f"Görüntü işlendi: {imagePath.name}")
    print(f"Sonuç kaydedildi: {outputPath}")

if __name__ == "__main__":
    main()