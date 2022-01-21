import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Create dataframe from given data
def read_data_df(size):
    data = pd.read_fwf('shift-masks.txt')
    d = data.iloc[1:, :]
    a = []
    for row in range(size):
        r = split_to_values(d.iloc[row])
        a.append(r)

    df = pd.DataFrame(a)
    return df


# Remove extra symbols and split each row to single int values
def split_to_values(row):
    arr_row = row.array
    rep1 = arr_row[0].replace("{", "")
    rep2 = rep1.replace("}", "")
    res = rep2.split(",")

    return res


# Create a PCA model for dimensionality reduction, fit and transform data
# Evaluate results by RMSE for the raw values and the binary values
def fit_pca(data, comp):
    pca = PCA(n_components=comp)
    data_tran = pca.fit_transform(data)
    data_restored = pca.inverse_transform(data_tran)

    rmse_raw = eval_rmse(data.astype(float), data_restored)

    binary_restored = find_shifts(data_restored)
    rmse_binary = eval_rmse(data.astype(float), binary_restored)

    return data_tran, rmse_raw, rmse_binary


# Make the restored data binary (1 for shift, 0 for no shit)
def find_shifts(data_r):
    for row in range(len(data_r)):
        for val in range(len(data_r[row])):
            if data_r[row,val] > 0.5:
                data_r[row,val] = 1.0
            else:
                data_r[row,val] = 0.0

    return data_r


# Evaluate results by RMSE
def eval_rmse(data_o, data_r):
    a = []
    for row in range(len(data_o)):
        d = data_o.iloc[row]
        rmse = math.sqrt(np.square(np.subtract(d, data_r[row])).mean())
        a.append(rmse)

    return a


if __name__ == '__main__':
    d_size = 1000 # How many shifts are used
    dataframe = read_data_df(d_size)

    n_comp = 30 # How many features the feature-space is reduced to
    red_data, rmse_r, rmse_b = fit_pca(dataframe[0:d_size], n_comp)

    # TODO: make graphs into single graph
    # figure, axis = plt.subplots(2, 2)
    
    # Plot binary results
    rmse_mean_b = sum(rmse_b) / len(rmse_b)
    plt.plot(rmse_b)
    plt.axhline(y=rmse_mean_b, c='r', ls='--')
    plt.xlabel('shift')
    plt.ylabel('RMSE')
    plt.title("Mean of RMSE for binary results %1.4f" % rmse_mean_b)
    plt.show()

    # Plot raw values
    rmse_mean = sum(rmse_r)/len(rmse_r)
    plt.plot(rmse_r)
    plt.axhline(y=rmse_mean, c='r', ls='--')
    plt.xlabel('shift')
    plt.ylabel('RMSE')
    plt.title("Mean of RMSE for raw results %1.4f" % rmse_mean)
    plt.show()
