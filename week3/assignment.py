import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sbn
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_ms
import sklearn.linear_model as sk_lm

""" This implements a test case for linear regression"""
""" Fit a linear model to the log of prostate-specific antigen lpsa, with other variables """


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


def prediction_stats(y_predicted, y_actual):
    mean_square_error = sk.metrics.mean_squared_error(y_predicted, y_actual)
    base_mean_square_error = sk.metrics.mean_squared_error(y_actual, [np.mean(y_actual)] * len(y_actual))
    result = dict()
    result["mean_square_error"] = mean_square_error
    result["r_square"] = 1 - mean_square_error / base_mean_square_error

    return result


if __name__ == '__main__':

    data = data_info("../dataset/prostate.csv")

    # data description of several variables
    """
    ◦ log cancer volume (lcavol)
    ◦ log prostate weight (lweight)
    ◦ age
    ◦ log of the amount of benign prostatic hyperplasia (lbph)
    ◦ seminal vesicle invasion (svi) – binary variable
    ◦ log of capsular penetration (lcp)
    ◦ Gleason score (gleason) – ordered categorical variable
    ◦ percent of Gleason scores (pgg45)
    ◦ log of prostate-specific antigen (lpsa)

    """
    # result of above function calls is that there are no missing values

    data.info()

    # since gleason is a categorical variable and hence converting into dummy variable
    data["gleason"] = data["gleason"].astype("object")
    data["svi"] = data["svi"].astype(int)

    # creating dummy variables for gleason
    data = pd.get_dummies(data)

    # plotting the correlation matrix
    corr = data.corr()
    # plt.figure(figsize=(12, 5))
    # sbn.heatmap(corr, annot=True)
    # plt.show()

    """
    From the above, we can see that lpsa( output variable) is highly correlated with lcavol, lcp, svi, lweight in decreasing orderr
    So in the final its weight should be a lot higher that others
    """

    input_var_list = list(set(list(data)) - set(["lpsa"]))
    output_var_list = ["lpsa"]

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

    print("why re-running does not create the same results as of prev run  results")
    pass














