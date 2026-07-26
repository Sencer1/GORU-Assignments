from pathlib import Path
# dosya ve klasör yollarını kullanmak için

import cv2
#  open cv  kütüphanesiniş eklemek için
# görüntü okuma görüntüyer çizilm yapma gibi işler için

player_class_id = 0
# veri setindeki oyuncuların class numarası

templateSayi = 10
templateBoyut = 256

trainImageDir = Path("data/original/train/images")

trainLabelDir = Path("data/original/train/labels")

templateOutputDir = Path("data/odevData/train/player")


# validation kısmı için burası

patchSize = 256

validationPosCount = 200 

validationNegCount = 600

validationImageDir = Path("data/original/valid/images")

validationLabelDir = Path("data/original/valid/labels")

validationPosDir = Path("data/odevData/validation/positive")

validationNegDir = Path("data/odevData/validation/negative")

validationOutputLabelDir = Path("data/odevData/validation/labels")



#  test kısmı için burası

testPosCount = 200

testNegCount = 600

testImageDir = Path("data/original/test/images")

testLabelDir = Path("data/original/test/labels")

testPosDir = Path("data/odevData/test/positive")

testNegDir = Path("data/odevData/test/negative")

testOutputLabelDir = Path("data/odevData/test/labels")


# yolo formatını dönüştürmek için bu fonksiyon
def yolo_to_pixel(centerX, centerY, width, height, imageWidth, imageHeight):

    centerX_pixel = centerX * imageWidth
    centerY_pixel = centerY * imageHeight

    width_Pixel = width * imageWidth
    height_Pixel = height * imageHeight

    #  burda resmin boyutuna göre pixel değerleirni aldık

    x1 = int(centerX_pixel - width_Pixel / 2)
    y1 = int(centerY_pixel - height_Pixel / 2)
    x2 = int(centerX_pixel + width_Pixel / 2)
    y2 = int(centerY_pixel + height_Pixel / 2)

    #  burda da kenar bilgilerini aldık orjinal resme göre

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(imageWidth - 1, x2)
    y2 = min(imageHeight - 1, y2)

    return x1, y1, x2, y2


# sadece futbolcu ısnıfa ait kutuları okumak için

def readPlayerBoxes(labelPath, imageWidth, imageHeight):

    playerBoxes = [] 

    # dosyayı açıp satır satır okuyoruz sonra da kapıtıyoruz with ile
    with labelPath.open("r", encoding="utf-8") as labelFile:
        for line in labelFile:

            values = line.strip().split()
            # strip baş ve sondaki gereksiz boşlukları siler
            # split metni boşluklardan ayırarak listeye dönüştürür

            if len(values) != 5:
                continue

            classId = int(values[0])

            # sadece oyuncu sınıfı için
            if classId != player_class_id:
                continue

            centerX = float(values[1])
            centerY = float(values[2])
            width = float(values[3])
            height = float(values[4])

            # yolo dan pixele dönüştürüyoruz burda

            playerBox = yolo_to_pixel(centerX, centerY, width, height, imageWidth, imageHeight)

            playerBoxes.append(playerBox)

    return playerBoxes



# oyuncuların kutu alanını hesaplamak için burası 

def calculateBoxArea(playerBox):
    
    x1, y1, x2, y2 = playerBox

    width = x2 - x1
    height = y2 - y1

    return width * height



# oyunucları resimde kırpmak için burası 

def cropPlayer(image, playerBox):

    x1, y1, x2, y2 = playerBox

    imageHeight, imageWidth = image.shape[:2]

    boxWidth = x2 - x1
    boxHeight = y2 - y1

    horizontalPadding = int(boxWidth * 0.05)
    verticalPadding = int(boxHeight * 0.05)

    # kırpılmış halinin kordinatlarını hesapliyoruz burda
    cropX1 = max(0, x1 - horizontalPadding)
    cropY1 = max(0, y1 - verticalPadding)

    # sınırları aşmasın diye kutuların genişletilmiş hali onu ayarlıyoruz
    cropX2 = min(imageWidth, x2 + horizontalPadding)
    cropY2 = min(imageHeight, y2 + verticalPadding)

    #  kırpmayı burda yapıyoruz

    playerCrop = image[cropY1:cropY2, cropX1:cropX2]

    return playerCrop


# burası ön işleme fonksiyonu gri ve 256 x 256 için

def preprocessTemplate(playerCrop):

    #  görüntü boş mu diye kontrol ettik burda
    if playerCrop.size == 0:
        return None
    
    # cvtColor ile renk değiştirmek için kullanıyoruz
    grayCrop = cv2.cvtColor(playerCrop, cv2.COLOR_BGR2GRAY)

    #  burası da boyutu ayarlamak için resmi verdik boyut bilgisini verdik hangi yöntem kullanılacak onu seçtik
    resizedCrop = cv2.resize(grayCrop, (templateBoyut, templateBoyut), interpolation=cv2.INTER_AREA)

    return resizedCrop


# kutuların kesişip kesişmediğini konrol için burası

def boxIntersect(firstBox, secondBox):

    fx1, fy1, fx2, fy2 = firstBox
    sx1, sy1, sx2, sy2 = secondBox

    return not (fx2 <= sx1 or fx1 >= sx2 or fy2 <= sy1 or fy1 >= sy2)


# oyuncuları resimden çıkarmak için positif örnek için

def createCenterPatchBox(playerBox, imageWidth, imageHeight):
    if (imageWidth < patchSize or imageHeight < patchSize):
        return None
    
    x1, y1, x2, y2 = playerBox

    playerCenterX = (x1 + x2) // 2
    playerCenterY = (y1 + y2) // 2

    # burası de yeni görüntünün resmin neresine denk gleiyor onu yazmak için

    patchX1 = playerCenterX - patchSize // 2
    patchY1 = playerCenterY - patchSize // 2

    patchX1 = max(0, min(patchX1, imageWidth - patchSize))
    patchY1 = max(0, min(patchY1, imageHeight - patchSize))

    patchX2 = patchX1 + patchSize
    patchY2 = patchY1 + patchSize

    return patchX1, patchY1, patchX2, patchY2


# oyuncu kutularını yeni kırpılmış resimdeki kordinatlara göre ayarlmak için burası

def convertBoxtoPatchCor(playerBox, patchBox):

    playerX1, playerY1, playerX2, playerY2 = playerBox
    patchX1, patchY1, patchX2, patchY2 = patchBox

    newX1 = max(playerX1, patchX1) - patchX1
    newY1 = max(playerY1, patchY1) - patchY1
    newX2 = min(playerX2, patchX2) - patchX1
    newY2 = min(playerY2, patchY2) - patchY1

    if (newX2 <= newX1 or newY2 <= newY1):
        return None
    
    return newX1, newY1, newX2, newY2



# burası pixel den yolo formatına çevirmek için

def pixelBoxToYolo(playerBox):

    x1, y1, x2, y2 = playerBox

    boxWidth = x2 - x1 
    boxHeight = y2 - y1

    centerX = x1 + boxWidth / 2
    centerY = y1 + boxHeight / 2

    norCenterX =  centerX / patchSize
    norCenterY = centerY / patchSize

    norWidth = boxWidth / patchSize
    norHeight = boxHeight / patchSize

    return norCenterX, norCenterY, norWidth, norHeight


# pozitif validation görüntüsü oluşturmak için burası

def createValidationPosImages():

    validationPosDir.mkdir(parents=True, exist_ok=True)

    validationOutputLabelDir.mkdir(parents=True, exist_ok=True)

    labelPaths = sorted(validationLabelDir.glob("*.txt"))

    positiveCount = 0

    for labelPath in labelPaths:
        if positiveCount >= validationPosCount:
            break

        imagePath = (validationImageDir / f"{labelPath.stem}.jpg")

        if not imagePath.exists():
            continue

        image = cv2.imread(str(imagePath))

        if image is None:
            continue

        imageHeight, imageWidth = image.shape[:2]

        playerBoxes = readPlayerBoxes(labelPath, imageWidth, imageHeight)

        for selectedPlayerBox in playerBoxes:
            if positiveCount >= validationPosCount:
                break

            patchBox = createCenterPatchBox(selectedPlayerBox, imageWidth, imageHeight)

            if patchBox is None:
                continue

            patchX1, patchY1, patchX2, patchY2 = patchBox 

            patch = image[patchY1: patchY2, patchX1:patchX2]

            if patch.shape[:2] != (patchSize, patchSize):
                continue

            patchPlayerBoxes = []

            for playerBox in playerBoxes:

                convertedBox = convertBoxtoPatchCor(playerBox, patchBox)

                if convertedBox is not None:
                    patchPlayerBoxes.append(convertedBox)

            if not patchPlayerBoxes:
                continue

            positiveCount += 1

            fileName = f"positive_{positiveCount:04d}"

            imageOutputPath = (validationPosDir / f"{fileName}.jpg")

            labelOutputPath = (validationOutputLabelDir / f"{fileName}.txt")

            # cv2.imwrite(str(imageOutputPath), patch)
            grayPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            onay = cv2.imwrite(str(imageOutputPath), grayPatch)

            if not onay:
                raise RuntimeError(f"Görüntü kaydedilemedi: {imageOutputPath}")

            with labelOutputPath.open("w", encoding="utf-8") as labelFile:
                for playerBox in patchPlayerBoxes:
                    centerX, centerY, boxWidth, boxHeight = pixelBoxToYolo(playerBox)

                    labelFile.write(f"{player_class_id} " f"{centerX:.6f} " f"{centerY:.6f} " f"{boxWidth:.6f} " f"{boxHeight:.6f}\n")
            
            print(f"Pozitif validation görüntüsü: {positiveCount}/{validationPosCount}")

    if positiveCount < validationPosCount:
        raise RuntimeError(f"Yeterli poizitif validation görüntüsü oluşturulamadı: {positiveCount}")
            



# Negatif validation görüntü oluşturmak için burası

def createValidationNegImages():

    validationNegDir.mkdir(parents=True, exist_ok=True)

    labelPaths = sorted(validationLabelDir.glob("*.txt"))

    negativeCount = 0

    stepSize = patchSize // 2

    for labelPath in labelPaths:
        if negativeCount >= validationNegCount:
            break

        imagePath = (validationImageDir / f"{labelPath.stem}.jpg")

        if not imagePath.exists():
            continue

        image = cv2.imread(str(imagePath))

        if image is None:
            continue

        imageHeight, imageWidth = image.shape[:2]

        playerBoxes = readPlayerBoxes(labelPath, imageWidth,imageHeight)


        for patchY1 in range(0,imageHeight - patchSize + 1, stepSize):
            if negativeCount >= validationNegCount:
                break

            for patchX1 in range(0, imageWidth - patchSize + 1, stepSize):
                if negativeCount >= validationNegCount:
                    break

                patchBox = (patchX1, patchY1, patchX1 + patchSize, patchY1 + patchSize)

                containsPlayer = any(boxIntersect(playerBox, patchBox) for playerBox in playerBoxes)

                if containsPlayer:
                    continue

                patch = image[patchY1:patchY1 + patchSize, patchX1:patchX1 + patchSize]

                if patch.shape[:2] != (patchSize, patchSize):
                    continue

                negativeCount += 1

                outputPath = ( validationNegDir / f"negative_{negativeCount:04d}.jpg")

                # cv2.imwrite(str(outputPath), patch)
                grayPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                onay = cv2.imwrite(str(outputPath), grayPatch)

                if not onay:
                    raise RuntimeError(f"Görüntü kaydedilemedi: {outputPath}")


                print(f"Negatif validation görüntüsü: {negativeCount}/{validationNegCount}")

    if negativeCount < validationNegCount:
        raise RuntimeError(f"Yeterli negatif validation görüntüsü oluşturulamadı: {negativeCount}")
    







#  train veri setindeki farklı görüntülerden birer oyunucu seçerek on eğitim şablonu  için burası 

def createPlayerTemplate():

    # klasörü oluşturmka için üst klasörler yoksa onları da açar ve varsa hata verme 
    templateOutputDir.mkdir(parents=True,exist_ok=True)

    # train klasöründeki tüm txt dosyalarını bulur glob ile sonra da onları sıraya koyar
    labelPaths = sorted(trainLabelDir.glob("*.txt"))


    # kaç tane şablon oluştuğunu syamak i,çin burası 
    templateCount = 0

    for labelPath in labelPaths:
        if templateCount >= templateSayi:
            break

        # .stem dosya adını uzantısız verir sonra bunları birleştiriyoruz
        imagePath = (trainImageDir / f"{labelPath.stem}.jpg")

        if not imagePath.exists():
            continue

        image = cv2.imread(str(imagePath))

        if image is None:
            continue

        imageHeight, imageWidth = image.shape[:2]

        #  oyuncuları okuyoruz burda
        playerBox = readPlayerBoxes(labelPath, imageWidth, imageHeight)

        if not playerBox:
            continue
        
        # listedeki en büyük elemnaı bulmak için yöntem olarak yazdığımız fonk kullanıyor
        largestPlayerBox = max(playerBox, key=calculateBoxArea) 

        x1, y1, x2, y2 = largestPlayerBox

        boxWidth = x2 - x1
        boxHeight = y2 - y1

        # çok küçük ve belirsiz olanları kullanmamak içn burası
        if boxWidth < 20 or boxHeight < 40:
            continue

        # kırpılmış oyuncu resmi 
        playerCrop = cropPlayer(image, largestPlayerBox)

        # oyunuc resmini 256 x 256 ve gri yaptık
        processedTemplate = preprocessTemplate(playerCrop)

        if preprocessTemplate is None:
            continue

        templateCount += 1

        # çıktının kaydedilecği tam dosya yolu için burası
        outputPath = (templateOutputDir / f"player_{templateCount:02d}.png")

        # write ile kaydediceğe pathe o görüntüyü yazar
        sonuc = cv2.imwrite(str(outputPath), processedTemplate)

        if not sonuc:
            raise RuntimeError(f"Şablon kaydedilemedi: {outputPath}")
        
        print(f"{templateCount}. şablon oluşturuldu: {imagePath.name} - {outputPath}")

    if templateCount < templateSayi:
        raise RuntimeError(f"{templateCount} şablon oluşturuldu")
    
    print(f"Toplam {templateCount} oyuncu şablonu oluşturuldu")

        


# pozitif test görüntülerini oluşturmak için

def createTestPosImages():

    testPosDir.mkdir(parents=True, exist_ok=True)

    testOutputLabelDir.mkdir(parents=True, exist_ok=True)

    labelPaths = sorted(testLabelDir.glob("*.txt"))

    positiveCount = 0

    for labelPath in labelPaths:
        if positiveCount >= testPosCount:
            break

        imagePath = (testImageDir / f"{labelPath.stem}.jpg")

        if not imagePath.exists():
            continue

        image = cv2.imread(str(imagePath))

        if image is None:
            continue

        imageHeight, imageWidth = image.shape[:2]

        playerBoxes = readPlayerBoxes(labelPath, imageWidth, imageHeight)
        
        for selectedPlayerBox in playerBoxes:
            if positiveCount >= testPosCount:
                break

            patchBox = createCenterPatchBox(selectedPlayerBox, imageWidth,imageHeight)

            if patchBox is None:
                continue

            patchX1, patchY1, patchX2, patchY2 = patchBox

            patch = image[patchY1:patchY2, patchX1:patchX2]

            if patch.shape[:2] != (patchSize, patchSize):
                continue

            patchPalyerBoxes = []

            for playerBox in playerBoxes:
                convertedBox = convertBoxtoPatchCor(playerBox, patchBox)

                if convertedBox is not None:
                    patchPalyerBoxes.append(convertedBox)

            if not patchPalyerBoxes:
                continue

            positiveCount += 1

            fileName = (f"positive_{positiveCount:04d}")

            imageOutputPath = (testPosDir / f"{fileName}.jpg")

            labelOutputPath = (testOutputLabelDir / f"{fileName}.txt")

            grayPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            onay = cv2.imwrite(str(imageOutputPath), grayPatch)

            if not onay:
                raise RuntimeError(f"Test görüntüsü kaydedilemedi: {imageOutputPath}")
            
            with labelOutputPath.open("w", encoding="utf-8") as labelFile:
                for playerBox in patchPalyerBoxes:
                    
                    centerX, centerY, boxWidth, boxHeight = pixelBoxToYolo(playerBox)

                    labelFile.write(f"{player_class_id} " f"{centerX:.6f} " f"{centerY:.6f} " f"{boxWidth:.6f} " f"{boxHeight:.6f}\n")

            print(f"Pozitif test görüntüsü: {positiveCount}/{testPosCount}")
    
    if positiveCount < testPosCount:
        raise RuntimeError(f"Yeterli pozitif test görüntüsü oluşmadı: {positiveCount}")
    


# negatif test görüntüler için burası da

def createTestNegImages():

    testNegDir.mkdir(parents=True, exist_ok=True)

    labelPaths = sorted(testLabelDir.glob("*.txt"))

    negativeCount = 0

    stepSize = patchSize // 2

    for labelPath in labelPaths:
        if negativeCount >= testNegCount:
            break

        imagePath = (testImageDir / f"{labelPath.stem}.jpg")

        if not imagePath.exists():
            continue

        image = cv2.imread(str(imagePath))

        if image is None:
            continue

        imageHeight, imageWidth = image.shape[:2]

        playerBoxes = readPlayerBoxes(labelPath, imageWidth,imageHeight)

        for patchY1 in range(0, imageHeight - patchSize + 1, stepSize):
            if negativeCount >= testNegCount:
                break

            for patchX1 in range(0, imageWidth - patchSize + 1, stepSize):
                if negativeCount >= testNegCount:
                    break

                patchBox = (patchX1, patchY1, patchX1 + patchSize, patchY1 + patchSize)

                containsPlayer = any(boxIntersect(playerBox, patchBox) for playerBox in playerBoxes)

                if containsPlayer:
                    continue

                patch = image[patchY1:patchY1 + patchSize, patchX1:patchX1 + patchSize]

                if patch.shape[:2] != (patchSize, patchSize):
                    continue

                negativeCount += 1

                outputPath = (testNegDir / f"negative_{negativeCount:04d}.jpg")

                grayPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                onay = cv2.imwrite(str(outputPath), grayPatch)

                if not onay:
                    raise RuntimeError(f"Test görüntüsü kaydedilemedi: {outputPath}")
                
                print(f"Negatif test görüntüsü: {negativeCount}/{testNegCount}")

    if negativeCount < testNegCount:
        raise RuntimeError(f"Yeterli negatif test görüntüsü oluşmadı: {negativeCount}")


def main():
    # # görütünün dosya yolunu oluşturmak için
    # imagePath = Path(r"C:\Users\USER\Desktop\468Odev\data\original\train\images\000005.jpg")

    # labelPath = Path(r"C:\Users\USER\Desktop\468Odev\data\original\train\labels\000005.txt")

    # # imread resmi okumak için str ile path nesnesi metne dönüşütürülür
    # image = cv2.imread(str(imagePath))

    # if image is None:
    #     raise FileNotFoundError(f"Görüntü okunamadı: {imagePath}")
    # #  görüntü okunazmsa hata atıyoruz raise hata atmak için

    # if not labelPath.exists():
    #     raise FileNotFoundError(f"Label dosyası bulunamadı: {labelPath}")
    
    # # yğksekliği ve genişliği aldık burda
    # imageHeight, imageWidth = image.shape[:2]
    
    # playerBoxes = readPlayerBoxes(labelPath, imageWidth, imageHeight)
    # # oyuncuları bulduk

    # print(f"Görüntü boyutu: {imageWidth}x{imageHeight}")
    # print(f"Bulunan oyuncu sayısı: {len(playerBoxes)}")

    # # her oyuncunun etrafına dikdörtgen çiziyoruz burda

    # for x1, y1, x2, y2 in playerBoxes:
    #     # dikdörtgen çiziyor bura
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     # resim sol üst ve sağ alt köşe renk kalınlık

    # outputPath = Path("playerBoxTest.jpg")

    # # write de yazar. yazacağı path i alır ve kaydedeceği görüntüyü alır
    # sonuc = cv2.imwrite(str(outputPath), image)

    # # kaydedemezsek hata atsın
    # if not sonuc:
    #     raise RuntimeError("Sonuç kaydedilemedi")
    
    # print(f"Sonuç Kaydedildi: {outputPath}")


    # # oyuncu template oluşturma
    createPlayerTemplate()

    # pozitif ve negatif validation görüntüsü oluşturma
    createValidationPosImages()
    createValidationNegImages()

    print("\nValidation veri seti oluşturuldu")

    # pozitif ve negatif test görüntüsü oluşturma
    createTestPosImages()
    createTestNegImages()

    print("\nTest veri seti oluşturuldu")


if __name__ == "__main__":
    main()