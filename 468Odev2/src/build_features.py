# burası train val ve test için featureları üretir

# yapılan işlemler 
# görüntü okur, gri tona çevirir, 256x256 yapar, 9 pencereye böler, her pencereye özellik çıkarmak, iou ile her pencereye etiket atar, featureları .npz olarak kaydeder

import numpy as np

from config import TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR,VAL_LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR, FEATURE_DIR

from dataset_utils import readAndPreprocessImage, readYoloLabels, createWindows, assignWindowLabel, findImageFiles

from hog_features import createHogDescriptor, extractWindowHogFeatures


# hog feature ve etiket çıkarma için

def buildSplitFeatures(imageDir, labelDir, outputPath, removeMultiObjectWindows):

    hog = createHogDescriptor()
    windows = createWindows()
    imagePaths = findImageFiles(imageDir)

    allFeatures = []
    allLabels = []
    allImageNames = []
    allWindowCor = []

    skippedWindows = 0

    for imagePath in imagePaths:
        labelPath = labelDir / f"{imagePath.stem}.txt"

        image = readAndPreprocessImage(imagePath)
        boxes = readYoloLabels(labelPath)

        for window in windows:
            label = assignWindowLabel(window, boxes, removeMultiObjectWindows)
            # görüntüedeki her pencere için boxlar ile kıyas yaptı label atadı

            if label is None:
                skippedWindows += 1
                continue

            features = extractWindowHogFeatures(image, window, hog)

            allFeatures.append(features)
            allLabels.append(label)
            allImageNames.append(imagePath.name)
            allWindowCor.append(window)

    featuresArray = np.array(allFeatures, dtype=np.float32)
    labelsArray = np.array(allLabels, dtype=np.int64)
    imageNamesArray = np.array(allImageNames)
    windowsArray = np.array(allWindowCor, dtype=np.int64)

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    # diskte daha az yer kaplasın diye sıkıştırır npz olarak saklar
    np.savez_compressed(outputPath, features=featuresArray, labels=labelsArray, image_names=imageNamesArray, window_coordinates=windowsArray)

    print(f"Kaydedildi: {outputPath}")
    print(f"Toplam pencere sayısı: {len(labelsArray)}")
    print(f"Atlanan çoklu nesne penceresi: {skippedWindows}")
    print(f"Özellik boyutu: {featuresArray.shape}")


def main():

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    buildSplitFeatures(TRAIN_IMAGE_DIR,TRAIN_LABEL_DIR, outputPath=FEATURE_DIR / "train_features.npz", removeMultiObjectWindows=True)

    buildSplitFeatures(VAL_IMAGE_DIR, VAL_LABEL_DIR, outputPath=FEATURE_DIR / "val_features.npz", removeMultiObjectWindows=True)

    buildSplitFeatures(TEST_IMAGE_DIR, TEST_LABEL_DIR, outputPath=FEATURE_DIR / "test_features.npz", removeMultiObjectWindows=False)



if __name__ == "__main__":
    main()