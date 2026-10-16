import numpy as np

class LinearRegressionVectorized:
    """Линейная регрессия: аналитическое решение через нормальное уравнение (вообще ноль циклов)"""
    def fit(self, X, y):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias


class LogisticRegressionVectorized:
    """Логистическая регрессия (векторизованный градиентный спуск, без циклов по данным)"""
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs): # Цикл по эпохам
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            dw = (X.T @ (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X, threshold=0.5):
        z = X @ self.weights + self.bias
        return (self._sigmoid(z) >= threshold).astype(int)


import numpy as np

class KNNVectorized:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        squared_distances = np.sum(diff ** 2, axis=2)

        # Находим индексы k ближайших соседей
        k_indices = np.argsort(squared_distances, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]

        predictions = []
        for row in k_labels:
            labels, counts = np.unique(row, return_counts=True)
            predictions.append(labels[np.argmax(counts)])

        return np.array(predictions)


class NaiveBayesVectorized:
    """Наивный Байес (векторизованный по матрицам)"""
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.sum(np.log(np.maximum(numerator / denominator, 1e-6)), axis=1)

    def predict(self, X):
        posteriors = [np.log(self.priors[c]) + self._pdf(c, X) for c in self.classes]
        posteriors = np.array(posteriors) # (n_classes, n_samples)
        return self.classes[np.argmax(posteriors, axis=0)]


class PerceptronVectorized:
    """Перцептрон без бэкпропа (векторизованный по эпохам)"""
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_output = X @ self.weights + self.bias
            y_pred = (linear_output >= 0).astype(int)

            error = y - y_pred
            self.weights += self.lr * (X.T @ error) / n_samples
            self.bias += self.lr * np.sum(error) / n_samples

    def predict(self, X):
        linear_output = X @ self.weights + self.bias
        return (linear_output >= 0).astype(int)
