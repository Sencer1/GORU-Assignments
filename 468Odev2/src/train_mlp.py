# burası mlp nin eğitimi için
# featureları yükler ve standardize etmek 
# class weight hesaplama
# pytorch ile mlp model eğitimi
# en iyi modeli kaydetmek


import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
# sklearn makine öğrenmesi için kullanılır veri ölçekleme
#  tran test split accruacy gibi metrikler ve klasik ml modelleri için

from sklearn.preprocessing import StandardScaler
# featurları standirdicze etmek için burası
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# veriyi modele düzenli vermek için burası
# dataloader batch batch vermek için
# tensordataset ile label feature modele düzgün vermek için


from config import FEATURE_DIR, MODEL_DIR, BEST_MODEL_PATH, NUM_CLASSES, RANDOM_SEED
from mlp_model import HOGMLP



# sonuçların tekrar üretilebilmesi için seed ayarlamak için
def setSeed(seed):
    random.seed(seed)
    # random modülünün rastgeleliği sabitlemek için
    # tekrar random aynı sonucu üretmeye çalışır
    np.random.seed(seed)
    # burası da np nin rastgeliliğini sabitlemek için
    torch.manual_seed(seed)
    # cpu da pytorch un rastgeleliğini sabitlemek için burası ilk ağırlıklar için
    torch.cuda.manual_seed_all(seed)
    # burası da cuda tarafı için sabitlemeye çalışır

# npz feature dosayasını yüklemek için
def loadFeatures(path):
    data = np.load(path)
    features = data["features"]
    labels = data["labels"]

    return features, labels


# sınıf dengesizliği için weight hesaplama ve
# veri sayısı az olan sınıfa daha yüksek ağırlık vermek için burası
def calculateClassWeights(labels):

    classCounts = np.bincount(labels, minlength=NUM_CLASSES)
    # o sınıftan kaç tane olduğunu tekrarladığını saymak için burası np.bincount
    # min length ile class sayısı kadar label sayısı kadar olsun mutlaka sonuç demiş oluyoruz

    totalCount = len(labels)

    weights = totalCount / (NUM_CLASSES * np.maximum(classCounts, 1))
    # az olan sınıfın ağırlığını arttırmak için burası
    # np.max güvenlik önelmi sınıf 0 ise 0 a bölünmesin diye
    
    # hesaplanan ağırlıkları tensor e çevirir
    # bu ağırlıklar daha sonra loss hesaplarken kullanılır 
    return torch.tensor(weights, dtype=torch.float32)

# val set için accuracy ve f1 hesaplamak için
def evaluateModel(model, dataloader, device):
    model.eval()

    allPredictions = []
    allLabels = []

    # gradient hesaplamadan sadece tahmin ve sonuç ölçmek için
    with torch.no_grad():
        for features, labels in dataloader:
            # burası veriyi cihaza taşımak için
            # model ve veri aynı yerde olmalı
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            
            # burası çıktıda her sonuç tensorü için o satırdaki en büyük değeri bulup indexi dönüyor
            # dim=1 satırlara bakmak için
            predictions = torch.argmax(outputs, dim=1)

            allPredictions.extend(predictions.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())

            # önce gpudaysa cpu ya alınır sonra numpy a çevirilir
            # extend ile listeye tek tek eklenir

    accuracy = accuracy_score(allLabels, allPredictions)
    macroF1 = f1_score(allLabels, allPredictions, average="macro", zero_division=0)
    # average macro her sınıf için f1 ayrı hesaplanır sonra ortlaması alınır önce player sonra background için f1
    # zero division da hiç player gelmezse mesela 0 a bölme sonucu direkt 0 kabul et demek

    return accuracy, macroF1


# tek hyperparameter ile model eğitimi için burası
def trainOneModel(trainFeatures, trainLabels, valFeatures, valLabels, hiddenSize, hiddenLayers, learningRate, batchSize, dropoutRate, useBatchNorm, patience, maxEpochs, device):
    # pateince earlt stop için değer
    # val f1 mesela 8 epoch boyunca iyileşmezse eğitimi durdur

    inputSize = trainFeatures.shape[1]
    # her örneğin feature sayısnı aldık

    scaler = StandardScaler()
    # sklearn den standardScaler nesnesini oluşturduk
    # bu nesne feature değerlerini standardize eder

    trainFeaturesScaled = scaler.fit_transform(trainFeatures)
    # her sutün için o featuralrın değerlerine göre ortlama ve stand sapma hesaplar
    # fit ile öğrendi transform ile featuralrı bu değerlere göre değşitirdi

    valFeaturesScaled = scaler.transform(valFeatures)
    # fit kullanımıyoruz çünkü traindeki verinin ort ve standart sapması ile yapmak istiyoruz bu ödnüşümü daha iyi öğrenme için

    trainDataset = TensorDataset(torch.tensor(trainFeaturesScaled, dtype=torch.float32), torch.tensor(trainLabels, dtype=torch.long))
    # train featureları ve labelları ile pytroch dataset haline getiriyor
    # numpy arrayden tensor e çeviriyor

    valDataset = TensorDataset(torch.tensor(valFeaturesScaled, dtype=torch.float32), torch.tensor(valLabels, dtype=torch.long))
    # aynı işlem val için de yaptık

    trainLoader = DataLoader(trainDataset, batchSize, shuffle=True)
    # burası ile veriyi batch batch ayırdık

    valLoader = DataLoader(valDataset, batchSize, shuffle=True)


    model = HOGMLP(inputSize, NUM_CLASSES, hiddenSize, hiddenLayers, useBatchNorm, dropoutRate).to(device)

    classWeights = calculateClassWeights(trainLabels).to(device)

    criterion = nn.CrossEntropyLoss(weight=classWeights)
    # class ağırlıklarını göre loss hesaplamak için burası

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # modelin ağırlıklaırnı güncellemk için burası adam optimizer a göre

    bestValF1 = 0.0
    bestState = None

    epochsWithputImprovement = 0

    for epoch in range(1, maxEpochs + 1):
        model.train()

        totalLoss = 0.0

        for features, labels in trainLoader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # önceki batchden gelen graidentleri sıfırlamka için

            outputs = model(features)
            loss = criterion(outputs,labels)

            loss.backward()
            # burda gradientları hesaplıyoruz
            optimizer.step()
            # burası da ağırlıkları günceller

            totalLoss += loss.item()
            # tensor ü normal python sayısına çevirmek için
        valAccuracy, valF1 = evaluateModel(model, valLoader, device)

        print(f"Epoch {epoch:03d} | Loss: {totalLoss / len(trainLoader):.4f} | Val Acc: {valAccuracy:.4f} | Val F1: {valF1:.4f}")

        if valF1 > bestValF1:
            bestValF1 = valF1
            bestState = model.state_dict()
            # modelin o anki ağırlıklaırnı saklamak içn
            epochsWithputImprovement = 0
        else:
            epochsWithputImprovement += 1

        if epochsWithputImprovement >= patience:
            print("Early stopping uygulandı.")
            break

    model.load_state_dict(bestState)
    return model, scaler, bestValF1


def main():

    setSeed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cuda varsa cuda da yoksa cpu da 

    print(f"Kullanılan cihaz: {device}")

    trainFeatures, trainLabels = loadFeatures(FEATURE_DIR / "train_features.npz")
    # burda feature ve labelları çekiyor

    valFeatures, valLabels = loadFeatures(FEATURE_DIR / "val_features.npz")

    # farklı hyperparametre denemeleri burası

    experiments = [
        {
            "hiddenSize" : 256,
            "hiddenLayers" : 2,
            "learningRate" : 0.001,
            "batchSize" : 64,
            "dropoutRate" : 0.30,
            "useBatchNorm" : True,
            "patience" : 15,
            "maxEpochs" : 60
        },
        {
            "hiddenSize" : 512,
            "hiddenLayers" : 2,
            "learningRate" : 0.001,
            "batchSize" : 64,
            "dropoutRate" : 0.40,
            "useBatchNorm" : True,
            "patience" : 12,
            "maxEpochs" : 60},
        {
            "hiddenSize" : 256,
            "hiddenLayers" : 3,
            "learningRate" : 0.0005,
            "batchSize" : 128,
            "dropoutRate" : 0.30,
            "useBatchNorm" : True,
            "patience" : 8,
            "maxEpochs" : 60
        },
        {
            "hiddenSize" : 512,
            "hiddenLayers" : 3,
            "learningRate" : 0.0005,
            "batchSize" : 128,
            "dropoutRate" : 0.40,
            "useBatchNorm" : True,
            "patience" : 10,
            "maxEpochs" : 70
        },
        {
            "hiddenSize" : 128,
            "hiddenLayers" : 2,
            "learningRate" : 0.001,
            "batchSize" : 64,
            "dropoutRate" : 0.20,
            "useBatchNorm" : True,
            "patience" : 20,
            "maxEpochs" : 60
        },
        {
            "hiddenSize" : 256,
            "hiddenLayers" : 2,
            "learningRate" : 0.0001,
            "batchSize" : 128,
            "dropoutRate" : 0.30,
            "useBatchNorm" : True,
            "patience" : 10,
            "maxEpochs" : 80
        },
        {
            "hiddenSize" : 512,
            "hiddenLayers" : 2,
            "learningRate" : 0.0001,
            "batchSize" : 128,
            "dropoutRate" : 0.50,
            "useBatchNorm" : True,
            "patience" : 15,
            "maxEpochs" : 80
        },
        {
            "hiddenSize" : 256,
            "hiddenLayers" : 3,
            "learningRate" : 0.001,
            "batchSize" : 64,
            "dropoutRate" : 0.40,
            "useBatchNorm" : False,
            "patience" : 8,
            "maxEpochs" : 60
        },
        {
            "hiddenSize" : 512,
            "hiddenLayers" : 6,
            "learningRate" : 0.0005,
            "batchSize" : 128,
            "dropoutRate" : 0.20,
            "useBatchNorm" : True,
            "patience" : 20,
            "maxEpochs" : 100
        },
        {
            "hiddenSize" : 512,
            "hiddenLayers" : 5,
            "learningRate" : 0.001,
            "batchSize" : 64,
            "dropoutRate" : 0.40,
            "useBatchNorm" : True,
            "patience" : 25,
            "maxEpochs" : 100
        }
    ]

    bestModel = None
    bestScaler = None
    bestParams = None
    bestScore = 0.0

    for index, params in enumerate(experiments, start=1):
        print("\n" + "=" * 60)
        print(f"Deney {index}")
        print(params)
        print("=" * 60)

        model, scaler, valF1 = trainOneModel(trainFeatures, trainLabels, valFeatures, valLabels, device=device, **params)


        print(f"Deney {index} validation F1: {valF1:.4f}")

        if valF1 > bestScore:
            bestScore = valF1
            bestModel = model
            bestScaler = scaler
            bestParams = params

        
    checkpoint = {
        "model_state_dict" : bestModel.state_dict(),
        "scaler" : bestScaler,
        "best_params" : bestParams,
        "best_val_f1" : bestScore,
        "input_size" : trainFeatures.shape[1],
        "num_classes" : NUM_CLASSES
    }

    torch.save(checkpoint, BEST_MODEL_PATH)
    # en iyi modeli kaydetmek için burası
    
    print("\n En iyi model kaydedildi:")
    print(BEST_MODEL_PATH)

    print("En iyi parametreler:")
    print(bestParams)

    print(f"En iyi validation F1: {bestScore:.4f}")


if __name__ == "__main__":
    main()