import pickle
# python nesnelerini dosyalara kaydetmek ve daha sonra aynı haliyle geri yüklemek için 
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from config import CLASS_COUNT, KMEANS_MAX_ITERATIONS, KMEANS_N_INIT, RANDOM_SEED

# tek node temsili için ktree de
class KTreeNode:

    def __init__(self, depth):
        self.depth = depth
        self.kmeans_model: KMeans | None = None
        self.children: dict[int, KTreeNode] = {}

        self.class_counts = np.zeros(CLASS_COUNT, dtype=np.int64)

# sift descriptorlarını hiyerarşik kMeans ile bölen ağaç burası
class KTree:
    def __init__(self, k_value, max_depth):

        self.k_value = k_value
        self.max_depth = max_depth

        self.root = KTreeNode(depth=0)

    # eğitimdeki descriptorları ile agacı oluşturmak için burası
    def fit(self, features, labels):

        if len(features) != len(labels):
            raise ValueError("Özellik ve etiket sayıları eşit değil.")

        if len(features) == 0:
            raise ValueError("Hiç özellik yok.")

        self.root = KTreeNode(depth=0)

        self._fit_node(self.root, features, labels)


    # bir düğümü eğitmek ve alt düğümleri oluşturmak için
    def _fit_node(self, node:KTreeNode, features, labels):

        node.class_counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.int64)

        if node.depth >= self.max_depth:
            return

        if len(features) < self.k_value:
            return

        unique_feature_count = len(np.unique(features, axis=0))

        cluster_count = min(self.k_value, len(features), unique_feature_count)

        if cluster_count < 2:
            return


        kmeans_model = KMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITERATIONS)

        cluster_labels = kmeans_model.fit_predict(features)

        node.kmeans_model = kmeans_model

        unique_cluster_ids = np.unique(cluster_labels)

        for cluster_id in unique_cluster_ids:
            cluster_mask = cluster_labels == cluster_id

            child_features = features[cluster_mask]
            child_labels = labels[cluster_mask]

            child_node = KTreeNode(depth=node.depth + 1)

            node.children[int(cluster_id)] = child_node

            self._fit_node(child_node, child_features, child_labels)




    # descriptorun ağaçtaki karşılık geldiği yaprağı bulmak için
    def find_leaf(self, descriptor):

        current_node = self.root

        while current_node.kmeans_model is not None:
            cluster_id = int(current_node.kmeans_model.predict(descriptor.reshape(1, -1))[0])

            if cluster_id not in current_node.children:
                break

            current_node = current_node.children[cluster_id]

        return current_node


    # bir bölgedeki descriptorların düştüğü yapraklardaki eğitim sınıflarının oylarını toplamak için toplam oy oranları sınıf olasılığı olarak kullanılıyor
    def predict_region_probabilities(self, descriptors):
        total_votes = np.zeros(CLASS_COUNT, dtype=np.float64)

        if len(descriptors) == 0:
            total_votes[0] = 1.0
            return total_votes

        for descriptor in descriptors:
            leaf = self.find_leaf(descriptor)

            # total_votes = total_votes + leaf.class_counts
            leaf_vote_sum = leaf.class_counts.sum()

            if leaf_vote_sum == 0:
                continue

            leaf_probabilities = leaf.class_counts / leaf_vote_sum

            total_votes = total_votes + leaf_probabilities

        vote_sum = total_votes.sum()

        if vote_sum == 0:
            total_votes[0] = 1.0
            return total_votes

        return total_votes / vote_sum



    # bölge için tahmin edilen sınıfını ve olasılığını döndürmek için
    def predict_region(self, descriptors):

        probabilities = self.predict_region_probabilities(descriptors)

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return predicted_class, confidence


    # yaprak bölgelerini temsil eden son kmeans merkezlerini toplmak için burası
    def collect_leaf_centers(self):

        centers = []

        self._collect_leaf_centers_recursive(self.root, centers)

        if not centers:
            return np.empty((0, 128), dtype=np.float32)

        return np.asarray(centers, dtype=np.float32)


    # yapraklara karşılık gelen küme merkezlerini toplamak için
    def _collect_leaf_centers_recursive(self, node:KTreeNode, centers):

        if node.kmeans_model is None:
            return

        for cluster_id, child_node in node.children.items():
            cluster_center = (node.kmeans_model.cluster_centers_[cluster_id])

            if child_node.kmeans_model is None:
                centers.append(cluster_center)
            else:
                self._collect_leaf_centers_recursive(child_node, centers)


    # burası modeli pickle dosyasına kaydetmek için
    def save(self, model_path):

        model_path.parent.mkdir(parents=True, exist_ok=True)

        with model_path.open("wb") as model_file:
            pickle.dump(self, model_file)
            # python nesnesini dosyaya kaydeder dump

        
    
    # kaydedilmiş modeli yüklemek için burası
    @staticmethod
    def load(model_path):

        with model_path.open("rb") as model_file:
            model = pickle.load(model_file)

        return model
            