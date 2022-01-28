import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.metrics import KLDivergence, CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam, SGD
from datetime import datetime


# Create dataframe from given data
def read_data_df(size):
    data = pd.read_fwf('shift-masks.txt')
    d = data.iloc[1:, :]
    a = []
    for row in range(size):
        r = split_to_values(d.iloc[row])
        a.append(r)

    df_new = pd.DataFrame(a)
    df_res = df_new.astype(float)
    return df_res


# Remove extra symbols and split each row to single int values
def split_to_values(row):
    arr_row = row.array
    rep1 = arr_row[0].replace("{", "")
    rep2 = rep1.replace("}", "")
    res = rep2.split(",")

    return res


# Create a PCA model for dimensionality reduction, fit and transform data
def fit_pca(data, comp):
    pca_model = PCA(n_components=comp)
    data_tran = pca_model.fit_transform(data)
    data_restored = pca_model.inverse_transform(data_tran)

    return pca_model, data_tran, data_restored


# Create a FA model for dimensionality reduction, fit and transform data
# TODO: find a way to inverse transform fa
def fit_fa(data, comp):
    factor_analyser = FactorAnalysis(n_components=comp, random_state=0)
    df_fa = factor_analyser.fit_transform(data)

    return factor_analyser, df_fa


# Create an auto_encoder for dimensionality reduction
# TODO: improve the performance and try different dimensionality reductions
def auto_encoder(train, val, dims, save_model=False, load_model=False):
    if load_model:
        loaded_model = tf.keras.models.load_model("auto_encoder")
        return loaded_model

    n_neurons_d = np.arange(25, 63, 7)
    n_neurons_e = np.flip(n_neurons_d)

    # Encoder
    encoder = Sequential(name="Encoder")
    encoder.add(Dense(dims, input_shape=[63], activation='relu'))
    # for n in range(len(n_neurons_e)):
        # encoder.add(Dense(n_neurons_e[n], activation='relu'))
    print(encoder.summary())

    # Decoder
    decoder = Sequential(name="Decoder")
    # for n in range(len(n_neurons_d)):
        # decoder.add(Dense(n_neurons_d[n], activation='relu'))   # len(n_neurons_d)-1
    decoder.add(Dense(63, input_shape=[dims], activation='sigmoid'))
    # print(decoder.summary())

    # Auto-encoder
    autoencoder = Sequential([encoder, decoder], name="Auto-encoder")
    print(autoencoder.summary())
    autoencoder.compile(loss="mse", optimizer=Adam(),
                        metrics=[BinaryAccuracy()])
    history = autoencoder.fit(train, train, batch_size=8, epochs=20, validation_data=(val, val), verbose=1)
    # SGD(learning_rate=0.001, momentum=0.9)

    if save_model:
        autoencoder.save("auto_encoder")

    return autoencoder, encoder, decoder, history


# Plot the loss and accuracy for training and validations sets
def plot_progress(hist):
    figure, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(hist.history['binary_accuracy'], label="training")
    ax[0].plot(hist.history['val_binary_accuracy'], label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")

    ax[1].plot(hist.history['loss'], label="training")
    ax[1].plot(hist.history['val_loss'], label="validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss")

    plt.legend()
    plt.show()


# Predict shifts based on auto-encoder
def encoder_predictor(encoder, decoder, val):
    coded_val = encoder.predict(val)
    restored_val = decoder.predict(coded_val)

    return restored_val


# Evaluate the chosen model based on kl divergence
def model_evaluation(orig, restored):
    res_bi = set_binary(restored)

    print(res_bi[0])

    kl = KLDivergence()
    kl.update_state(orig, res_bi)
    kl_val = kl.result().numpy()

    return kl_val


# Set restored values to binary data (1 for shift 0 if not)
def set_binary(val):
    for i in range(len(val)):
        for j in range(len(val[i])):
            if val[i, j] < 0.2:
                val[i, j] = 0
            else:
                val[i, j] = 1
    return val


# Count distribution of data
def count_distribution(data):
    lst = [0]*63
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i, j] == 1:
                lst[j] = lst[j] + 1
    print(lst)
    return lst


if __name__ == '__main__':
    startTime = datetime.now()  # For measuring execution time
    d_size = 1000000  # How many shifts are used
    n_comp = 25  # How many features the feature-space is reduced to
    encoder_decoder = True  # True if auto-encoder is initialized and trained
    benchmarks = False  # True if benchmarking methods are initialized and trained

    # Read data
    dataframe = read_data_df(d_size)

    # split data for training, validation, and testing
    train_set, test_set = train_test_split(dataframe.values, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.1)

    # print(test_set[0:3])

    # TODO: run on GPU --> make sure cuda installation works
    if encoder_decoder:

        # Define and fit auto_encoder
        auto_en, enc, dec, auto_hist = auto_encoder(train_set, val_set, n_comp)

        # Evaluate results
        restored_data = encoder_predictor(enc, dec, test_set)
        kl_value = model_evaluation(test_set, restored_data)

        print("KL divergence: ", kl_value)

        print("Execution time: ", datetime.now() - startTime)

        plot_progress(auto_hist)

    elif benchmarks:
        dim_red = np.arange(5, 60, 5)
        res = []
        for d in dim_red:
            # Fit transform to pca
            pca, data_pca, data_pca_res = fit_pca(train_set, d)

            test_d = pca.transform(test_set)
            restored_test = pca.inverse_transform(test_d)

            # Evaluate results
            kl_value = model_evaluation(test_set, restored_test)
            res.append(kl_value)

        # fa, data_fa = fit_fa(train_set, n_comp)

        plt.plot(dim_red, res)
        plt.xlabel("dimensions")
        plt.ylabel("KL divergence")
        plt.show()

        print("KL divergence: ", res)

        print("Execution time: ", datetime.now() - startTime)
