import numpy as np
from sklearn.utils import check_X_y
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans




class BaseSMOTE:
    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)

    def _compute_samples_needed(self, y):
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        n_min = class_counts[minority_class]
        n_maj = class_counts[majority_class]

        if self.sampling_strategy == 'auto':
            n_samples = n_maj - n_min
        else:
            desired_min = int(self.sampling_strategy * n_maj)
            n_samples = max(0, desired_min - n_min)

        return minority_class, majority_class, n_samples

    def _interpolate(self, x_i, x_nn):
        lam = self.rng_.random()
        return x_i + lam * (x_nn - x_i)




class SMOTEGenerator(BaseSMOTE):
    def __init__(self, k_neighbors=5, sampling_strategy='auto', random_state=None):
        super().__init__(sampling_strategy, random_state)
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        minority_class, _, n_samples = self._compute_samples_needed(y)

        if n_samples == 0:
            return X, y

        X_min = X[y == minority_class]
        n_min = len(X_min)

        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_min))
        nn.fit(X_min)
        neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        synthetic = []
        for _ in range(n_samples):
            idx = self.rng_.integers(0, n_min)
            x_i = X_min[idx]
            x_nn = X_min[self.rng_.choice(neighbors[idx])]
            synthetic.append(self._interpolate(x_i, x_nn))

        X_syn = np.array(synthetic)
        y_syn = np.full(len(X_syn), minority_class)

        return np.vstack((X, X_syn)), np.hstack((y, y_syn))
    



class ClusterSMOTEGenerator(BaseSMOTE):
    def __init__(self, k_neighbors=5, n_clusters=3, sampling_strategy='auto', random_state=None):
        super().__init__(sampling_strategy, random_state)
        self.k_neighbors = k_neighbors
        self.n_clusters = n_clusters

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        minority_class, _, n_samples_total = self._compute_samples_needed(y)

        if n_samples_total == 0:
            return X, y

        X_min = X[y == minority_class]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_min)

        synthetic = []
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        ratios = counts / counts.sum()
        samples_per_cluster = (ratios * n_samples_total).astype(int)

        for cid, n_samples in zip(unique_clusters, samples_per_cluster):
            X_cluster = X_min[cluster_labels == cid]
            n_cluster = len(X_cluster)

            if n_cluster <= 1 or n_samples == 0:
                continue

            k_eff = min(self.k_neighbors, n_cluster - 1)
            nn = NearestNeighbors(n_neighbors=k_eff + 1)
            nn.fit(X_cluster)
            neighbors = nn.kneighbors(X_cluster, return_distance=False)[:, 1:]

            for _ in range(n_samples):
                idx = self.rng_.integers(0, n_cluster)
                x_i = X_cluster[idx]
                x_nn = X_cluster[self.rng_.choice(neighbors[idx])]
                synthetic.append(self._interpolate(x_i, x_nn))

        if not synthetic:
            return X, y

        X_syn = np.array(synthetic)
        y_syn = np.full(len(X_syn), minority_class)

        return np.vstack((X, X_syn)), np.hstack((y, y_syn))