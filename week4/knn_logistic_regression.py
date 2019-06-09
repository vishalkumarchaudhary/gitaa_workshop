""" This is week4 assignment on K- nearest neighbour and logistic regression"""

import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


def data_info(file_name):

    data_df = pd.read_csv(file_name)
    print('Header of file {} are {}'.format(file_name, list(data_df)))

    print(data_df.describe(include="all"))

    print("correlation matrix is")
    correlation_mat = data_df.corr()
    print(correlation_mat)

    sbn.heatmap(correlation_mat, annot=True)
    plt.show()
    return data_df


if __name__ == '__main__':
    data = data_info("../dataset/crashTest.csv")
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data.replace({"Hatchback": 0, "SUV": 1}, inplace=True)

    # sbn.pairplot(data, hue="CarType")
    # plt.plot()

    print("As expected, there was no correlation between variables but we can see the intl and "+
          "HVACi with car type is somewhat correlated")

    # extracting features and output variables
    feature_list = list(set(list(data)) - set(["CarType"]))
    output_list = list(["CarType"])
    features = data[feature_list]
    output = data[output_list]

    # train test split on data
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=.3)

    # normalising the train data and then fitting both train and test data
    scaler = sk.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("K-NN model")

    """ KNN model initialisation and prediction for several values of k"""

    mis_classification = []
    for neighbours in range(2, 12):
        knn = KNeighborsClassifier(n_neighbors=neighbours)
        knn.fit(X_train, y_train)
        y_hat = knn.predict(X_test)
        mis_classification.append((y_hat != y_test.values.T).sum())

    plt.plot([i for i in range(2, 12)], mis_classification)
    plt.xlabel("Value of k")
    plt.ylabel("mis classified points")
    plt.title("Results on test data normalised on train data")
    plt.show()
    #
    # best_k = np.argmin(mis_classification)
    # knn = KNeighborsClassifier(n_neighbors=best_k)
    # knn.fit(X_train, y_train)
    # y_hat = knn.predict(X_test)
    # print(" k = {} is optimal as test error is minimum at this point".format(best_k))
    # print("Test error is {}".format((y_hat != y_test.values.T).sum()))

    print("Logistic regression")
    """ Logistic regression model and its predictions"""
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_hat = lr.predict(X_test)
    mis_classed = (y_hat != y_test.values.T).sum()
    print("mis classified on test data are ", mis_classed)

    print("Result : Logistic regression is performing better than the KNN (k=3 best k ) ")
