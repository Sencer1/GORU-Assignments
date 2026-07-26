# mpl modelini tanımlamak için burası
# mmodelin görevi 7560 feature ı alıp sınıfı tahmin etmek



import torch
from torch import nn

class HOGMLP(nn.Module):

    def __init__(self, inputSize, numClasses, hiddenSize=256, hiddenLayers=2, useBatchNorm=True, dropoutRate=0.30):

        super().__init__()

        # burası katmanlardaki işleri sırayla tutmak için
        layers = []
        # burası layerlar arası geçerken input sizeları güncellemke için
        currentSize = inputSize

        for turn in range(hiddenLayers):

            layers.append(nn.Linear(currentSize, hiddenSize))

            if useBatchNorm:
                layers.append(nn.BatchNorm1d(hiddenSize))

            layers.append(nn.ReLU())

            if dropoutRate > 0:
                layers.append(nn.Dropout(dropoutRate))

            currentSize = hiddenSize

        layers.append(nn.Linear(currentSize, numClasses))

        # burası bu pyhton listesini pytorch model akışına çevirmek için
        self.network = nn.Sequential(*layers)  


    def forward(self, x):
        return self.network(x)
