import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('D:\IISc\Sem2\ML\Assignments\A1\SVM_Assignment1\data\mnist_train.csv')
    test_df = pd.read_csv('D:\IISc\Sem2\ML\Assignments\A1\SVM_Assignment1\data\mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    X_train_norm = X_train / 255.0 * 2 - 1
    X_test_norm = X_test / 255.0 * 2 - 1
    return X_train_norm, X_test_norm


def plot_metrics(metrics) -> None:
    # plot and save the results
    k_values = [m[0] for m in metrics]
    accuracy_values = [m[1] for m in metrics]
    precision_values = [m[2] for m in metrics]
    recall_values = [m[3] for m in metrics]
    f1_score_values = [m[4] for m in metrics]

    # plot the metrics
    plt.plot(k_values, accuracy_values, label='accuracy')
    plt.plot(k_values, precision_values, label='precision')
    plt.plot(k_values, recall_values, label='recall')
    plt.plot(k_values, f1_score_values, label='f1-score')

    # set the x-axis and y-axis labels
    plt.xlabel('Number of principal components')
    plt.ylabel('Performance metric value')

    # set the plot title
    plt.title('Multi-class SVM Performance metrics vs. number of principal components')

    # set the legend
    plt.legend()

    # save the plot
    plt.savefig('performance_metrics.png')
    
    # grid view
    plt.grid()

    # show the plot
    plt.show()