import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

RMSE_list = []
RMSE_list_val = []
pred_list = []
pred_list_val = []

#############################
#       Read data           #
#############################


# Read person, score, and shifts from data
def read_personal_data(person):
    data = pd.read_fwf(person)
    a = []
    for row in range(len(data)):
        r = split_data(data.iloc[row].values)
        a.append(r)

    df_new = pd.DataFrame(a)
    return df_new


# Split personal data into values
def split_data(row):
    arr_row = str(row)
    arr_row = arr_row.replace("[", "")
    arr_row = arr_row.replace("]", "")
    arr_row = arr_row.replace("'", "")
    result = arr_row.split(",")
    return result


# Get dataframes for each person scoring shifts
def get_personal_data(data):
    data, num_people = get_num_people(data)
    dfs = []
    for i in range(num_people):
        df = data.loc[data['Person_category'] == i]
        lst = df.index[df['Score'] == '-1'].tolist()
        df = df.drop(lst)
        dfs.append(df)
    return dfs


# Get number of different people who have answered
def get_num_people(data):
    data.rename(columns={1: 'Person', 0: 'Score'}, inplace=True)
    data['Person'] = data['Person'].astype('category')
    data['Person_category'] = data['Person'].cat.codes

    num_people, cols = data.groupby('Person').count().shape

    return data, num_people


#################################
#       Create Bayesian NN      #
#################################

def gaussian_layer(x):
    '''
    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    mu, sigma = tf.unstack(x, num=2, axis=-1)

    # Add one dimension to make the right shape
    mu = tf.expand_dims(mu, -1)
    sigma = tf.expand_dims(sigma, -1)

    # Apply a softplus to make positive
    mu = tf.keras.activations.sigmoid(mu)
    sigma = tf.keras.activations.sigmoid(sigma)

    # Join back together again
    out_tensor = tf.concat((mu, sigma), axis=num_dims - 1)

    return out_tensor
    '''

    """
    Lambda function for generating normal distribution parameters
    mu (mean) and sigma (variance) from a Dense(2) output.
    Assumes tensorflow 2 backend.

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)

    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer

    Returns
    -------
    out_tensor : tf.Tensor

    """

    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    n, p = tf.unstack(x, num=2, axis=-1)

    # Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # Apply a softplus to make positive
    n = tf.keras.activations.softplus(n)

    # Apply a sigmoid activation to bound between 0 and 1
    p = tf.keras.activations.sigmoid(p)

    # Join back together again
    out_tensor = tf.concat((n, p), axis=num_dims - 1)

    return out_tensor


def negative_binomial_loss(y_true, y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.

    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    # Separate the parameters
    n, p = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # Calculate the negative log likelihood
    nll = (
            tf.math.lgamma(n)
            + tf.math.lgamma(y_true + 1)
            - tf.math.lgamma(n + y_true)
            - n * tf.math.log(p)
            - y_true * tf.math.log(1 - p)
    )

    '''
    n = len(y_pred)
    val = 2
    n = tf.cast(n, tf.float32)
    tf.cast(sigma, tf.float32)
    val = tf.cast(val, tf.float32)
    tf.cast(math.pi, tf.float32)

    print(type(val), type(math.pi), type(n))

    calc_loss = - (n / val) * tf.math.log(val * math.pi)
                # - (n / val) * tf.math.log(sigma**val) \
                # - (1 / (val*sigma**val))*sum(y_true-mu)
    '''

    return nll


def bayesian_nn(x_train, y_train):
    model = keras.Sequential(name="BayesianPP")

    probability_layer = Lambda(gaussian_layer)

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(probability_layer)

    model.compile(loss=negative_binomial_loss, optimizer=Adam(), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=50, verbose=0)

    print(model.summary())

    return model, history


# Compile and fit pp model that is trained from scratch
def scratch_model(x_train, y_train):
    model = Sequential(name="PersonalPreferenceModel2")

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=50, verbose=0)

    print(model.summary())

    return model, history


def general_probability(x_tr, y_tr, save=False):
    unc = np.zeros(len(y_tr)).flatten()
    val_split = int(0.9*len(x_tr))
    inputs = Input(shape=(63, ), name='input')

    x = Dense(512, activation='relu')(inputs)
    x = Dense(2048, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)

    # Head 1 (predicted score)
    x = Dense(512, activation='relu', name='learning_pred')(x)
    z1 = Dense(1, activation='sigmoid', name='prediction')(x)

    # Head 2 (error of prediction)
    x = Dense(512, activation='relu', name='learning_error')(x)
    z2 = Dense(1, activation='sigmoid', name='error')(x)

    model = Model(inputs=inputs, outputs=[z1, z2])
    model.compile(loss={'prediction': 'mean_squared_error', 'error': 'mean_squared_error'},
                  optimizer=Nadam(),
                  loss_weights={'prediction': 1, 'error': 1},
                  metrics={'prediction': 'mae', 'error': 'mae'})
    training = CustomTraining(np.array(x_tr[:val_split]),
                              (np.array(y_tr[:val_split]).flatten(), np.array(unc[:val_split])),
                              np.array(x_tr[val_split:]),
                              (np.array(y_tr[val_split:]).flatten(), np.array(unc[val_split:])),
                              model, 64)
    history = model.fit(training, epochs=50, verbose=1)    # , callbacks=[uncertainty])

    if save:
        model.save("probability_general")

    return model, history


def prob_NN(X_train, y_train):
    loaded_model = tf.keras.models.load_model("probability_general")
    y_pred = loaded_model.predict(X_train)
    unc = y_pred[1].flatten()
    val_split = int(0.9*len(X_train))

    inputs = Input(shape=(63, ), name='inputs')

    # Freeze some layers and add to model
    for layer in loaded_model.layers[:3]:
        layer.trainable = False

    # Check the trainable status of layers
    for layer in loaded_model.layers:
        print(layer, layer.trainable)

    inputs = loaded_model.layers[0](inputs)
    print("inputs: ", inputs)
    x = loaded_model.layers[1](inputs)
    print("x: ", x)
    x = loaded_model.layers[2](x)
    print("x again: ", x)

    '''
    # Head 1 (predicted score)
    x = Dense(512, activation='relu', name='learning_pred')(inputs)
    z1 = Dense(1, activation='sigmoid', name='prediction')(x)

    # Head 2 (error of prediction)
    x = Dense(512, activation='relu', name='learning_error')(inputs)
    z2 = Dense(1, activation='sigmoid', name='error')(x)

    model = Model(inputs=inputs, outputs=[z1, z2])
    '''
    # Head 1 (predicted score)
    x1 = loaded_model.layers[3](x)
    print("x1: ", x1)
    x2 = loaded_model.layers[4](x)
    print("x2: ", x2)

    # Head 2 (error of prediction)
    y1 = loaded_model.layers[5](x1)
    print("y1: ", y1)
    y2 = loaded_model.layers[6](x2)
    print("y2: ", y2)

    model = Model(inputs=inputs, outputs=[y1, y2])

    model.compile(loss={'prediction': 'mean_squared_error', 'error': 'mean_squared_error'},
                  optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                  loss_weights={'prediction': 1, 'error': 1},
                  metrics={'prediction': 'mae', 'error': 'mae'})
    training = CustomTraining(np.array(X_train[:val_split]),
                              (np.array(y_train[:val_split]).flatten(), np.array(unc[:val_split])),
                              np.array(X_train[val_split:]),
                              (np.array(y_train[val_split:]).flatten(), np.array(unc[val_split:])),
                              model, 64)
    history = model.fit(training, epochs=100, verbose=1)    # , callbacks=[uncertainty])

    return model, history


class CustomTraining(Sequence):
    def __init__(self, x_train, y_train, x_val, y_val, model, batch_size):
        self.x_train = x_train
        self.y_train = y_train[0]
        self.unc = y_train[1]
        self.x_val = x_val
        self.y_val = y_val[0]
        self.unc_val = y_val[1]
        self.model = model
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x_train) / self.batch_size)

    def __getitem__(self, idx):
        X = self.x_train[idx * self.batch_size:(idx + 1) *
          self.batch_size]
        y = self.y_train[idx * self.batch_size:(idx + 1) *
          self.batch_size]
        u = self.unc[idx * self.batch_size:(idx + 1) *
          self.batch_size]
        return X, (y, u)

    def on_epoch_end(self):
        y, u = self.model.predict(self.x_train)
        y = y.flatten()
        self.y_train = self.y_train.flatten()
        res = np.absolute(np.subtract(self.y_train, y))
        self.unc = res

        mse = mean_squared_error(self.unc, u.flatten())
        rmse = np.sqrt(mse)
        RMSE_list.append(rmse)

        mse_y = mean_squared_error(self.y_train, y)
        rmse_y = np.sqrt(mse_y)
        pred_list.append(rmse_y)

        y_v, u_v = self.model.predict(self.x_val)
        y_v = y_v.flatten()
        self.y_val = self.y_val.flatten()
        res_val = np.absolute(np.subtract(self.y_val, y_v))
        self.unc_val = res_val

        mse_val = mean_squared_error(self.unc_val, u_v.flatten())
        rmse_val = np.sqrt(mse_val)
        RMSE_list_val.append(rmse_val)

        mse_val_y = mean_squared_error(self.y_val, y_v.flatten())
        rmse_val_y = np.sqrt(mse_val_y)
        pred_list_val.append(rmse_val_y)


# Check for rows that are exactly the same, remove one of those
def check_same(dataf):
    counter = 0
    same_shifts = []
    data = dataf.values
    for i in range(len(data)):
        for j in range(len(data)):
            if j != i:
                if (data[i, :62] == data[j, :62]).all():
                    counter += 1
                    print("SAME")
                    same = [i, j]
                    same2 = [j, i]
                    if (same not in same_shifts) and (same2 not in same_shifts):
                        same_shifts.append(same)
                        # dataf = dataf.drop([dataf.index[j]])
    return counter, same_shifts, dataf


# Compile and fit pp model (using pre-trained model and re-training it)
def personal_preference_model(x_train, y_train):
    loaded_model = tf.keras.models.load_model("g_scoring_model_new")

    # Freeze some layers
    for layer in loaded_model.layers[:3]:
        layer.trainable = False

    # Check the trainable status of layers
    for layer in loaded_model.layers:
        print(layer, layer.trainable)

    model = Sequential(name="PersonalPreferenceModel")
    model.add(loaded_model)

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=50, verbose=0)

    print(model.summary())

    return model, history


# Plot the predictions against real values
def plot_predictions(predicted, actual, vals):
    # vals = 21
    figure, ax = plt.subplots(1, 1, figsize=(7, 5))
    major_ticks = np.arange(0, vals, 1)
    error = predicted[:][1]
    predictions = predicted[:][0]

    pred = np.array(predictions[:vals]).flatten()
    err = np.array(error[:vals]).flatten()
    # print(pred.shape, err.shape)

    y1 = np.subtract(pred, err)
    y2 = np.add(pred, err)

    x = np.arange(vals).flatten()
    # print(y1.shape, y2.shape, x.shape)
    # print(type(y1), type(y2), type(x))

    ax.fill_between(x, y1, y2, color="lightgray")
    ax.plot(predictions[:vals], 'or', label="predicted2")
    ax.plot(actual[:vals], 'x', label="real")  # actual.values[:vals]
    ax.set_title("Predicted vs real scores of shifts")
    ax.set_xticks(major_ticks)
    ax.grid(which="major", linestyle='--')
    ax.set_xlim(-1, vals)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    d_size = 5000
    d_s = 2928113
    n_worst = 10
    kmeans_creation = False
    bins = 4
    runs = 20

    ###############################
    #      Read general data      #
    ###############################

    '''
    df = pd.read_csv('cleaned_df')

    model_, history_ = prob_NN(df.iloc[:d_size, :63], df.iloc[:d_size, 63])
    print(history_.history.keys())
    preds = model_.predict(df.iloc[d_size:d_size+30, :63])
    act = df.iloc[d_size:d_size+30, 63]

    plot_predictions(preds, act, vals=21)
    '''

    # Read personal data
    p_data = read_personal_data('answers.txt')
    p_dfs = get_personal_data(p_data)
    pers_data = p_dfs[2]  # Choose which personal data to use

    # pers_data = pers_data.iloc[4:74, :]
    print(pers_data)

    # Get shifts and their score
    p_shifts = pers_data.iloc[:, 2:65].apply(pd.to_numeric)
    p_score = np.array(pers_data['Score'].apply(pd.to_numeric))

    # Scale score to a scale 0-1
    p_scaler = MinMaxScaler()
    p_scores = p_scaler.fit_transform(p_score.reshape(-1, 1))

    actual_err = []
    pred_err = []
    for _ in range(runs):
        # Split data into train and test
        pX_train, pX_test, py_train, py_test = train_test_split(p_shifts.values, p_scores, test_size=0.15, shuffle=True)

        model_pp, history_pp = prob_NN(pX_train, py_train)
        preds = model_pp.predict(pX_test)

        pred_error = preds[:][1]
        actual_error = np.absolute(preds[:][0] - py_test)

        ind = np.arange(len(pred_error))
        dataframe2 = pd.DataFrame({'Index': ind.flatten(),
                                   'Pred_error': pred_error.flatten(), 'Actual_error': actual_error.flatten()})
        df_p_error = dataframe2.sort_values(by=['Pred_error'])
        df_a_error = dataframe2.sort_values(by=['Actual_error'])
        print(df_p_error)
        print(df_a_error, len(df_a_error))

        nr_values = int(len(df_a_error) / bins)
        bin_means_a = []
        bin_means_p = []
        for i in range(0, len(df_a_error), nr_values):
            vals_bin_a = df_p_error['Actual_error'].iloc[i:i + nr_values]
            vals_bin_p = df_p_error['Pred_error'].iloc[i:i + nr_values]
            bin_means_a.append(mean(vals_bin_a))
            bin_means_p.append(mean(vals_bin_p))
        actual_err.append(bin_means_a)
        pred_err.append(bin_means_p)

    print(actual_err[0])
    print(actual_err[1])
    print(np.array(actual_err).T[0])
    print(actual_err[0][0], actual_err[0][1], actual_err[1][0])

    mean_a = []
    mean_p = []
    for i in range(bins):
        mean_a.append(mean(np.array(actual_err).T[i]))
        mean_p.append(mean(np.array(pred_err).T[i]))

    plt.figure(1)
    plt.plot(np.array(actual_err).T, color='darkorange', alpha=0.2)
    plt.plot(mean_a, color='black', label='Mean of actual error')
    plt.legend()
    plt.title('Actual error')
    plt.grid()

    plt.figure(2)
    plt.plot(np.array(pred_err).T, color='darkorange', alpha=0.2)
    plt.plot(mean_p, color='black', label='Mean of predicted error')
    plt.title('Predicted error')
    plt.legend()
    plt.grid()
    plt.show()

    '''
    avg_pred = sum(pred_error) / len(pred_error)
    avg_actual = sum(actual_error) / len(actual_error)

    # Real ordered worsts
    pred_worst = df_p_error['Pred_error'].iloc[-n_worst:]
    actual_worst = df_a_error['Actual_error'].iloc[-n_worst:]
    avg_pred_worst = sum(pred_worst) / len(pred_worst)
    avg_actual_worst = sum(actual_worst) / len(actual_worst)

    # Ordered by the other error's worsts
    pred_worst_r = df_a_error['Pred_error'].iloc[-n_worst:]
    actual_worst_r = df_p_error['Actual_error'].iloc[-n_worst:]
    avg_pred_worst_r = sum(pred_worst_r) / len(pred_worst_r)
    avg_actual_worst_r = sum(actual_worst_r) / len(actual_worst_r)

    print(int(0.9*len(py_train)), len(py_test))
    print("Average of predicted errors: ", avg_pred)
    print("Average of actual errors: ", avg_actual)
    print("Average of", n_worst, "worst predicted errors (ordered by predicted worsts): ", avg_pred_worst)
    print("Average of", n_worst, "worst actual errors (ordered by actual worsts): ", avg_actual_worst)
    print("Average of", n_worst, "worst predicted errors (ordered by actual worsts): ", avg_pred_worst_r)
    print("Average of", n_worst, "worst actual errors (ordered by predicted worsts): ", avg_actual_worst_r)

    # pe = df_p_error['Index'].iloc[-5:]
    # ae = df_a_error['Index'].iloc[-5:]
    # print(pe, ae)
    '''

    plt.plot(RMSE_list, label="training")
    plt.plot(RMSE_list_val, label="validation")
    plt.title("RMSE of error during training")
    plt.legend()
    plt.show()

    plt.plot(pred_list, label="training")
    plt.plot(pred_list_val, label="validation")
    plt.title("prediction power during training")
    plt.legend()
    plt.show()

    '''
    results = np.absolute(pred_error - actual_error)
    mse = mean_squared_error(pred_error, actual_error)
    rmse = np.sqrt(mse)
    print(rmse)
    
    plt.plot(results)
    plt.show()
    '''
    plot_predictions(preds, py_test, vals=len(py_test))

    if kmeans_creation:
        df = pd.read_csv('cleaned_df')
        wcss = []
        for i in range(1, 100, 10):
            kmeans = KMeans(i)
            kmeans.fit(df.iloc[:, :63].values)
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)

        number_clusters = range(1, 100, 10)
        plt.plot(number_clusters, wcss)
        plt.title('Elbow test for clustering')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
