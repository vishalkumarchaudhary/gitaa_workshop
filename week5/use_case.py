import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sbn
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_ms
import sklearn.linear_model as sk_lm


def prediction_stats(y_predicted, y_actual):
    mean_square_error = sk.metrics.mean_squared_error(y_predicted, y_actual)
    base_mean_square_error = sk.metrics.mean_squared_error(y_actual, [np.mean(y_actual)] * len(y_actual))
    result = dict()
    result["mean_square_error"] = mean_square_error
    result["r_square"] = 1 - (mean_square_error / base_mean_square_error)

    result["sk_r_square"] = sk.metrics.r2_score(y_actual, y_predicted)

    return result


if __name__ == '__main__':
    data = pd.read_csv("../input/sandman.csv")
    print(data.describe())
    # no missing values except [Acidity]

    print("Number of datapoints having Nan in acidity column is {}".format(data["ACIDITY"].isna().sum()))

    print(data["ACIDITY"].mean(), data["ACIDITY"].median(), data["ACIDITY"].mode())

    # impuding the median value for missing nan values
    data["ACIDITY"].fillna(data["ACIDITY"].median(), inplace=True)

    # using lambda function for replacing nan
    data2 = data["ACIDITY"].apply(func=(lambda x: data["ACIDITY"].median() if np.isnan(x) else x))
    data2.head(5)

    # dropping the date column
    data.drop("Date", inplace=True, axis=1)

    corr = data.corr()
    plt.figure(figsize=(10, 10))
    sbn.heatmap(corr, annot=True)
    plt.show()

    # plotting the pair plot
    plt.figure(figsize=(13, 13))
    sbn.pairplot(data)
    plt.show()

    # finding the input/output variable list
    input_var_list = list(set(list(data)) - set(["TOTAL_DEFECT_PERCENTAGE"]))
    output_var_list = ["TOTAL_DEFECT_PERCENTAGE"]

    # converting all types to float type
    data = data.astype(float)

    # test train split
    train_x, test_x, train_y, test_y = sk_ms.train_test_split(data[input_var_list],
                                                              data[output_var_list],
                                                              test_size=.3)

    #  normalising data w.r.t training set only.
    scaler_x = sk.preprocessing.StandardScaler()
    scaler_y = sk.preprocessing.StandardScaler()

    # calculating normalising variable
    scaler_x.fit(train_x)
    scaler_y.fit(train_y)

    # mapping to new dimensional space
    normalised_train_x = scaler_x.transform(train_x)
    normalised_test_x = scaler_x.transform(test_x)

    normalised_train_y = scaler_y.transform(train_y)
    normalised_test_y = scaler_y.transform(test_y)

    """ Linear regression model creation"""
    lr = sk_lm.LinearRegression()
    lr.fit(normalised_train_x, normalised_train_y)

    # prediction with the linear model
    y_hat_normalised = lr.predict(normalised_test_x)
    y_hat = scaler_y.inverse_transform(y_hat_normalised)

    print("predicted stats are ", prediction_stats(y_hat, test_y))

    coefficients = [i for i in zip(lr.coef_[0], list(train_x))]

    # sorting and printing in decreasing order
    print(sorted(coefficients)[::-1])
    #  as  said above with the correlation, it gives the same weights as predicted
    #  lcavol, svi, lweight has higher weight but lcp does not have because lcp and lcavol has .68 correlation



