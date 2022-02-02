import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.metrics import KLDivergence, CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam, SGD
from datetime import datetime
EPSILON = 0.00001


# Create dataframe from given data
def read_data_df(size):
    data = pd.read_fwf('shift-masks.txt')
    d_f = data.iloc[1:, :]
    a = []
    for row in range(size):
        r = split_to_values(d_f.iloc[row])
        a.append(r)

    df_new = pd.DataFrame(a)
    df_res = df_new.astype(float)
    return df_res


# Remove extra symbols and split each row to single int values
def split_to_values(row):
    arr_row = row.array
    rep1 = arr_row[0].replace("{", "")
    rep2 = rep1.replace("}", "")
    result = rep2.split(",")

    return result


# Create a PCA model for dimensionality reduction, fit and transform data
def fit_pca(data, comp):
    pca_model = PCA(n_components=comp)
    data_tran = pca_model.fit_transform(data)
    data_restored = pca_model.inverse_transform(data_tran)

    return pca_model, data_tran, data_restored


# Create a FA model for dimensionality reduction, fit and transform data
def fit_fa(data, comp):
    factor_analyser = FactorAnalysis(n_components=comp, random_state=0)
    df_fa = factor_analyser.fit_transform(data)
    fa_loadings = factor_analyser.components_
    df_restored = np.matmul(df_fa, fa_loadings)

    return factor_analyser, df_fa, df_restored


# Create an ICA model for dim red, fit and transform data
# TODO: make ICA converge by increasing max_iter or decreasing tolerance
def fit_ica(data, comp):
    ica_model = FastICA(n_components=comp, max_iter=1000)
    df_ica = ica_model.fit_transform(data)
    df_res = ica_model.inverse_transform(df_ica)

    return ica_model, df_ica, df_res


# Create an auto_encoder for dimensionality reduction
def auto_encoder(train, val, dims, save_model=False, load_model=False):
    if load_model:
        loaded_model = tf.keras.models.load_model("auto_encoder")
        return loaded_model

    # n_neurons_d = np.arange(25, 63, 10)
    # n_neurons_e = np.flip(n_neurons_d)

    # Encoder
    encoder = Sequential(name="Encoder")
    encoder.add(Dense(dims, input_shape=[63], activation='relu'))
    # encoder.add(Dense(dims, activation='relu'))
    # for n in range(len(n_neurons_e)):
    # encoder.add(Dense(n_neurons_e[n], activation='relu'))
    print(encoder.summary())

    # Decoder
    decoder = Sequential(name="Decoder")
    # for n in range(len(n_neurons_d)):
    # decoder.add(Dense(n_neurons_d[n], activation='relu'))   # len(n_neurons_d)-1
    decoder.add(Dense(63, input_shape=[dims], activation='sigmoid'))
    # decoder.add(Dense(63, activation='sigmoid'))
    # print(decoder.summary())

    # Auto-encoder
    autoencoder = Sequential([encoder, decoder], name="Auto-encoder")
    print(autoencoder.summary())
    autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=0.001),
                        metrics=[BinaryAccuracy()])
    history = autoencoder.fit(train, train, batch_size=64, epochs=15, validation_data=(val, val), verbose=1)
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


# Evaluate the chosen model based on kl divergence and Jaccard index
def model_evaluation(orig, restored):
    res_bi = set_binary(restored)

    # print(res_bi[0])

    kl = KLDivergence()
    kl.update_state(orig, restored)
    kl_val = kl.result().numpy()

    j_score = 0
    for i in range(len(orig)):
        j_score = j_score + jaccard_score(orig[i], res_bi[i])
    j_val = j_score/len(orig)

    return kl_val, j_val


# Self-implemented version of Jaccard index
def model_eval_jaccard(orig, restored):
    res_bi = set_binary(restored)
    m11 = 0
    m01 = 0
    m10 = 0

    jaccard = []
    for i in range(len(orig)):
        for j in range(len(orig[i])):
            if (orig[i, j] == 1) and (res_bi[i, j] == 1):
                m11 = m11 + 1
            if (orig[i, j] == 0) and (res_bi[i, j] == 1):
                m01 = m01 + 1
            if (orig[i, j] == 1) and (res_bi[i, j] == 0):
                m10 = m10 + 1

        jaccard.append(m11/(m01 + m10 + m11))
    return sum(jaccard)/len(jaccard)


# Self-implemented version of KL divergence
def kl_divergence(orig, restored):
    t_d = np.array(count_distribution(orig))
    r_d = np.array(count_distribution(restored))

    t_distr = t_d/len(orig)
    r_distr = r_d/len(orig)

    if t_distr.all() == 0:
        kl_diver = np.sum(t_distr * np.log(t_distr / EPSILON))
    else:
        kl_diver = np.sum(r_distr * np.log(r_distr / t_distr))

    if kl_diver < 0:
        kl_diver = np.sum(t_distr * np.log(t_distr / r_distr))

    return kl_diver


# Self-implemented version of Hamming distance measure
def hamming_distance(orig, restored):
    o = np.array(orig)
    r = np.array(restored)
    ham_dist = []
    for i in range(len(orig)):
        ham_dist.append(sum(abs(o[i]-r[i])))
    ham_res = sum(np.array(ham_dist))
    ham_part = ham_res/(len(orig)*len(orig[0]))
    return ham_res, ham_part


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
    return lst


if __name__ == '__main__':
    startTime = datetime.now()  # For measuring execution time
    d_size = 1000000         # How many shifts are used
    n_comp = 25             # How many features the feature-space is reduced to
    encoder_decoder = True  # True if auto-encoder is initialized and trained
    benchmarks = False      # True if benchmarking methods are initialized and trained
    pca = False            # True for only training and testing pca

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
        restored_data = encoder_predictor(enc, dec, test_set)

        # Old evaluation method
        kl_old, jaccard_old = model_evaluation(test_set, restored_data)

        # New evaluation methods
        kl_div = kl_divergence(test_set, set_binary(restored_data))
        j_ind = model_eval_jaccard(test_set, restored_data)
        hamm_val, hamm_shift = hamming_distance(test_set, set_binary(restored_data))

        print("KL divergence (pre-implemented): ", kl_old)
        print("KL divergence (self-implemented): ", kl_div)
        print("Jaccard index (pre-implemented): ", jaccard_old)
        print("Jaccard index (self-implemented): ", j_ind)
        print("Hamming distance: ", hamm_val)
        print("Hamming distance/shifts: ", hamm_shift)
        print("Execution time: ", datetime.now() - startTime)

        distr_true = count_distribution(test_set)
        distr_pred = count_distribution(set_binary(restored_data))
        plt.plot(distr_true, label='true')
        plt.plot(distr_pred, label='pred')
        plt.legend()
        plt.show()

        plot_progress(auto_hist)

    elif benchmarks:
        dim_red = np.arange(10, 61, 10)
        kl_res = []
        j_res = []
        for d in dim_red:
            # Fit transform to pca and/or fa, ica
            pca, data_pca, data_pca_res = fit_pca(train_set, d)
            # fa, data_fa, data_fa_res = fit_fa(train_set, d)
            # ica, data_ica, data_ica_res = fit_ica(train_set, d)

            test_d = pca.transform(test_set)
            restored_test = pca.inverse_transform(test_d)

            # Evaluate results
            kl_value, j_value = model_evaluation(test_set, restored_test)

            kl_res.append(kl_value)
            j_res.append(j_value)

        # plt.plot(dim_red, kl_res, label='KL divergence')
        plt.plot(dim_red, j_res, label='Jaccard index')
        plt.xlabel("Dimensions")
        plt.ylabel("Evaluation")
        plt.legend()
        plt.show()

        print("KL divergence: ", kl_res)

        print("Execution time: ", datetime.now() - startTime)

    elif pca:

        pca, data_pca, data_pca_res = fit_pca(train_set, n_comp)
        test_d = pca.transform(test_set)
        restored_test = pca.inverse_transform(test_d)

        # Old evaluation method
        kl_old, jaccard_old = model_evaluation(test_set, restored_test)

        # New evaluation methods
        kl_div = kl_divergence(test_set, set_binary(restored_test))
        j_ind = model_eval_jaccard(test_set, restored_test)
        hamm_val, hamm_shift = hamming_distance(test_set, set_binary(restored_test))

        print("KL divergence (pre-implemented): ", kl_old)
        print("KL divergence (self-implemented): ", kl_div)
        print("Jaccard index (pre-implemented): ", jaccard_old)
        print("Jaccard index (self-implemented): ", j_ind)
        print("Hamming distance: ", hamm_val)
        print("Hamming distance/shifts: ", hamm_shift)
        print("Execution time: ", datetime.now() - startTime)

        distr_true = count_distribution(test_set)
        distr_pred = count_distribution(set_binary(restored_test))
        plt.plot(distr_true, label='true')
        plt.plot(distr_pred, label='pred')
        plt.legend()
        plt.show()
