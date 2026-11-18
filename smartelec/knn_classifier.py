"""
SmartElec K-Nearest Neighbors (KNN) Implementation
Includes both a pure NumPy implementation from first principles and an
optimized Scikit-Learn wrapper. This demonstrates theoretical mastery for interviews
and algorithmic flexibility for production.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

try:
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PureNumpyKNN:
    """
    Pure NumPy Implementation of K-Nearest Neighbors Classifier.
    Computes Euclidean distance without external machine learning dependencies.
    """

    def __init__(self, n_neighbors: int = 5, distance_threshold: float = 4500.0):
        self.k = n_neighbors
        self.distance_threshold = distance_threshold
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PureNumpyKNN":
        """
        Fit the model using X as training data and y as target labels.
        In KNN, 'training' simply involves memorizing the training dataset (lazy learning).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for features, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length ({len(X)} vs {len(y)})")

        self.X_train = X
        self.y_train = y
        return self

    def _compute_distances_single(self, query_point: np.ndarray) -> np.ndarray:
        """
        Computes Euclidean distance between a single query vector and all training vectors.
        Formula: d(p, q) = sqrt( sum( (p_i - q_i)^2 ) )
        """
        # Vectorized Euclidean distance calculation using NumPy broadcasting
        diff = self.X_train - query_point
        distances = np.linalg.norm(diff, axis=1)
        return distances

    def predict_single(self, query_point: np.ndarray) -> Tuple[str, float, bool]:
        """
        Predict label for a single feature vector.
        Returns:
            Tuple of (predicted_label, average_k_distance, is_confident)
        """
        if self.X_train is None or self.y_train is None or len(self.X_train) == 0:
            return "Unregistered", float("inf"), False

        distances = self._compute_distances_single(query_point)

        # Determine effective k (cannot exceed number of training samples)
        k = min(self.k, len(self.X_train))

        # Find indices of the k smallest distances
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        # Majority vote
        vote_counts = Counter(k_nearest_labels)
        predicted_label, _ = vote_counts.most_common(1)[0]
        avg_distance = float(np.mean(k_nearest_distances))

        # Check if the closest match is within the acceptable threshold
        is_confident = avg_distance <= self.distance_threshold
        if not is_confident:
            return "Unknown", avg_distance, False

        return str(predicted_label), avg_distance, True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch predict labels for multiple feature vectors."""
        X = np.asarray(X, dtype=np.float32)
        predictions = []
        for query in X:
            label, _, _ = self.predict_single(query)
            predictions.append(label)
        return np.array(predictions)


class FaceKNNClassifier:
    """
    Unified KNN Classifier supporting both 'numpy' and 'sklearn' engines.
    Provides fallback to NumPy if scikit-learn is not available.
    """

    def __init__(self, n_neighbors: int = 5, engine: str = "numpy", distance_threshold: float = 4500.0):
        self.n_neighbors = n_neighbors
        self.engine = engine.lower()
        self.distance_threshold = distance_threshold
        self.is_fitted = False

        if self.engine == "sklearn" and not SKLEARN_AVAILABLE:
            self.engine = "numpy"

        if self.engine == "sklearn":
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
        else:
            self.model = PureNumpyKNN(n_neighbors=n_neighbors, distance_threshold=distance_threshold)

        self.labels_: Optional[np.ndarray] = None
        self.features_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaceKNNClassifier":
        """Fits the underlying KNN model."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.features_ = X
        self.labels_ = y

        if self.engine == "sklearn":
            effective_k = min(self.n_neighbors, len(X))
            self.model.set_params(n_neighbors=effective_k)
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, query_vector: np.ndarray) -> Tuple[str, float, bool]:
        """
        Predicts label for a query vector.
        Returns:
            Tuple of (label, distance_score, is_confident)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been trained yet. Register voters first.")

        query_flat = np.asarray(query_vector, dtype=np.float32).flatten()

        if self.engine == "numpy":
            return self.model.predict_single(query_flat)
        else:
            # For sklearn, we also calculate distance for threshold validation
            distances, indices = self.model.kneighbors([query_flat])
            pred_label = self.model.predict([query_flat])[0]
            avg_distance = float(np.mean(distances[0]))
            is_confident = avg_distance <= self.distance_threshold

            if not is_confident:
                return "Unknown", avg_distance, False
            return str(pred_label), avg_distance, True
