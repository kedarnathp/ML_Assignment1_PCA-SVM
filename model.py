import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm



class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X: np.ndarray) -> None:
        # fit the PCA model
        # compute the mean and the eigenvectors
        self.mean = X.mean(axis=0)
        X = X - self.mean
        cov = np.dot(X.T, X) / X.shape[0]
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        self.eigenvectors = eigvecs[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # transform the data
        # project the data onto the principal components
        X = X - self.mean
        return np.dot(X, self.eigenvectors)
    
    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)



class SupportVectorModel:
    def __init__(self):
        self.w = None
        self.b = None

    def _initialize(self, X):
        # initialize the parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

    def fit(
            self, X, y, learning_rate: float, num_iters: int, C: float = 1.0
    ):
        self._initialize(X)

        # fit the SVM model using stochastic gradient descent

        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            idx = np.random.randint(len(X))
            x_i, y_i = X[idx], y[idx]
            # calculate the hinge loss and its gradient
            hinge_loss = max(0, 1 - y_i * (np.dot(x_i, self.w) + self.b))
            grad_w = -C * y_i * x_i if hinge_loss > 0 else 0
            grad_b = -C * y_i if hinge_loss > 0 else 0
            # add the regularization term only to the gradient of w
            grad_w += self.w
            # update the weights and bias using the learning rate and the gradient
            # self.w = self.w - learning_rate * grad_w / i
            # self.b = self.b - learning_rate * grad_b / i
            
            self.w = self.w - learning_rate * grad_w 
            self.b = self.b - learning_rate * grad_b 


    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.sign(np.dot(X, self.w) + self.b)

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)




class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        # self.num_classes = num_classes
        # self.models = []
        # for i in range(self.num_classes):
        #     self.models.append(SupportVectorModel())
        self.num_classes = num_classes
        self.models = [SupportVectorModel() for _ in range(self.num_classes)]    

 
    
    def _preprocess_data(self, X, y, C):
        X_new = []
        y_new = []
        for i in range(self.num_classes):
            X_new.append(X)
            y_new.append(np.where(y == i, 1, -1))
        return X_new, y_new

    

    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        X, y = self._preprocess_data(X, y, kwargs['C'])
        # then train the 10 SVM models using the preprocessed data for each class
        for i in range(self.num_classes):
            self.models[i].fit(X[i], y[i], **kwargs)
    
   



    # def predict(self, X) -> np.ndarray:
    #     # pass the data through all the 10 SVM models and return the class with the highest score
    #     scores = []
    #     for i in range(self.num_classes):
    #         scores.append(np.dot(X, self.models[i].w) + self.models[i].b)
    #     return np.argmax(scores, axis=0)

    def predict(self, X) -> np.ndarray:
        scores = np.array([np.dot(X, self.models[i].w) + self.models[i].b for i in range(self.num_classes)])
        return np.argmax(scores, axis=0)
    
    #added this method for confusion matrix
    def _confusion_matrix(self, y_true, y_pred):
        num_classes = self.num_classes
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        return cm
    
    
    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        cm = self._confusion_matrix(y, y_pred)

        class_precisions = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            class_precisions[i] = tp / (tp + fp) if tp + fp > 0 else 0

        return np.average(class_precisions, weights=np.bincount(y))

    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        cm = self._confusion_matrix(y, y_pred)

        class_recalls = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            class_recalls[i] = tp / (tp + fn) if tp + fn > 0 else 0

        return np.average(class_recalls, weights=np.bincount(y))

    def f1_score(self, X, y) -> float:
        precision = self.precision_score(X, y)
        recall = self.recall_score(X, y)
        return 2 * precision * recall / (precision + recall)

    
