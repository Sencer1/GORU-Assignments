#  burası eğitilen en iyi modelin test verisinde değerlendirmke için

# test için hog yükler
# best modeli yükler
# karmaşıklık matrisi çıkartma
# değerlendirme metrikleri ile değerlendirme


import numpy as np
import torch

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
# metriklkeri özel tablo olarak göstermek için class report 

from config import FEATURE_DIR, BEST_MODEL_PATH, RESULT_DIR
from mlp_model import HOGMLP

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Kullanılan cihaz: {device}")

    testData = np.load(FEATURE_DIR / "test_features.npz")

    testFeatures = testData["features"]
    # npz den featureları yükledik
    testLabels = testData["labels"]
    # npz den label ları yükledik

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    # sadece model ağırlıklaırnı değil diğer bilgileri de alıyor

    scaler = checkpoint["scaler"]

    testFeaturesScaled = scaler.transform(testFeatures)

    testTensor = torch.tensor(testFeaturesScaled, dtype=torch.float32).to(device)

    model = HOGMLP(inputSize=checkpoint["input_size"], numClasses=checkpoint["num_classes"], hiddenSize=checkpoint["best_params"]["hiddenSize"], hiddenLayers=checkpoint["best_params"]["hiddenLayers"], useBatchNorm=checkpoint["best_params"]["useBatchNorm"], dropoutRate=checkpoint["best_params"]["dropoutRate"]).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    # burda model ağırlıklarını yüklüyoruz
    model.eval()

    with torch.no_grad():
        outputs = model(testTensor)

        predictions = torch.argmax(outputs, dim=1).cpu().numpy()


    accuracy = accuracy_score(testLabels, predictions)
    precision = precision_score(testLabels, predictions, average="macro", zero_division=0)
    recall = recall_score(testLabels, predictions, average="macro", zero_division=0)
    f1 = f1_score(testLabels, predictions, average="macro", zero_division=0)

    confMatrix = confusion_matrix(testLabels, predictions)

    print("\n Test sonuçları")

    print("-" * 40)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")

    print("\n Karmaşıklık matrisi")
    print(confMatrix)

    print("\n Classification report")
    print(classification_report(testLabels, predictions, zero_division=0))

    resultPath = RESULT_DIR / "test_metrics.txt"

    with resultPath.open("w", encoding="utf-8") as file:
        file.write("Test Sonuçları \n")
        file.write("-" * 40 + "\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 score: {f1:.4f}\n")
        file.write("Karmaşıklık matrisi\n")
        file.write(str(confMatrix))
        file.write("\n\n Classification report\n")
        file.write(classification_report(testLabels, predictions, zero_division=0))

    print(f"Sonuçlar Kaydedildi: {resultPath}")


if __name__ == "__main__":
    main()