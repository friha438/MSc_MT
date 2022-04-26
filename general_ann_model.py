import pandas as pd
import numpy as np
import math
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_histogram
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.utils import Sequence

RMSE_list = []
RMSE_list_val = []
pred_list = []
pred_list_val = []


# TODO: Create a class for defining fictional personal preferences
#####################################
#      Night shift preferences      #
#####################################

# Get a list that only displays night shifts
def get_night_shifts(data):
    nights = []
    for i in range(len(data)):
        n_shifts = []
        for j in range(2, len(data[i]), 3):
            n_shifts.append(data[i, j])
        nights.append(n_shifts)
    return nights


# Count how many night shifts each person has
def count_nights(data):
    num_nights = np.array(get_night_shifts(data))
    n_nights = []
    for i in range(len(num_nights)):
        n_nights.append(np.sum(num_nights[i]))

    return n_nights


# Create a distribution of night shifts
def night_distribution(data):
    n_nights = count_nights(data)
    night_hist = np.histogram(n_nights, bins=np.arange(11))
    night_dist = rv_histogram(night_hist)

    return night_dist, n_nights


# Create a person to score good on many nights and bad on few
def night_person(dist, data):
    n_nights = count_nights(data)
    static_score = dist.cdf(n_nights)
    score = np.random.normal(loc=static_score, scale=0.005)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


# Create a person to score good on few nights and bad on many
def non_night_person(dist, data):
    n_nights = count_nights(data)
    static_score_night = dist.cdf(n_nights)
    score = abs(np.array(static_score_night) - 1)
    # score = np.random.normal(loc=static_score, scale=0.005)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


#######################################
#      Weekend shift preferences      #
#######################################

# Count how many days during the weekends a person works during a roster
def count_weekends(data):
    weekend_shifts = []
    for i in range(len(data)):
        counter = 0
        for j in range(14, 20, 1):
            if data[i, j] == 1:
                counter = counter + 1
        for j in range(35, 41, 1):
            if data[i, j] == 1:
                counter = counter + 1
        for j in range(56, 62, 1):
            if data[i, j] == 1:
                counter = counter + 1
        weekend_shifts.append(counter)
    return weekend_shifts


# Find the distribution of worked weekend shifts
def weekend_distribution(data):
    weekend_shifts = count_weekends(data)
    weekend_hist = np.histogram(weekend_shifts, bins=np.arange(11))
    weekend_dist = rv_histogram(weekend_hist)
    return weekend_dist, weekend_shifts


# Create a person that will score high if there are many weekend shifts
def weekend_person(dist, data):
    weekend_shifts = count_weekends(data)

    static_score = dist.cdf(weekend_shifts)
    score = np.random.normal(loc=static_score, scale=0.005)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


# Create a person that scores low if they have many weekend shifts
def non_weekend_person(dist, data):
    weekend_shifts = count_weekends(data)

    static_score_w = dist.cdf(weekend_shifts)
    score = abs(np.array(static_score_w) - 1)
    # score = np.random.normal(loc=static_score, scale=0.005)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


#########################################
#       Create model and evaluate       #
#########################################

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.history = None
        self.logs = None
        self.batch = None
        self.epoch = None

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
            self.logs = logs
        self.history = {'loss': [], 'val_loss': []}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.batch = batch
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch = epoch
        self.history['val_loss'].append(logs.get('val_loss'))


def general_probability_model(x_tr, y_tr, save_model=False):
    inputs = Input(shape=(63, ), name='input')

    x = Dense(512, activation='relu')(inputs)
    x = Dense(2048, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', name='prediction')(x)

    model = Model(inputs=inputs, outputs=outputs)

    history = LossHistory()

    model.compile(loss='mse',
                  optimizer=Nadam(learning_rate=0.001), metrics=MeanAbsoluteError())
    model.fit(x_tr, y_tr, batch_size=64, epochs=1,
              validation_split=0.1, verbose=1, callbacks=[history])

    print(model.summary())

    if save_model:
        model.save("prop_gen")

    return model, history


def general_probability(x_tr, y_tr, save=False):
    unc = np.zeros(len(y_tr)).flatten()
    val_split = int(0.9*len(x_tr))
    inputs = Input(shape=(63, ), name='input')

    x = Dense(512, activation='relu')(inputs)
    x = Dense(2048, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)

    # Head 1 (predicted score)
    x1 = Dense(512, activation='relu', name='learning_pred')(x)
    y1 = Dense(1, activation='sigmoid', name='prediction')(x1)

    # Head 2 (error of prediction)
    x2 = Dense(512, activation='relu', name='learning_error')(x)
    y2 = Dense(1, activation='sigmoid', name='error')(x2)

    model = Model(inputs=inputs, outputs=[y1, y2])
    model.compile(loss={'prediction': 'mean_squared_error', 'error': 'mean_squared_error'},
                  optimizer=Nadam(),
                  loss_weights={'prediction': 1, 'error': 1},
                  metrics={'prediction': 'mae', 'error': 'mae'})
    training = CustomTraining(np.array(x_tr[:val_split]),
                              (np.array(y_tr[:val_split]).flatten(), np.array(unc[:val_split])),
                              np.array(x_tr[val_split:]),
                              (np.array(y_tr[val_split:]).flatten(), np.array(unc[val_split:])),
                              model, 64)
    history = model.fit(training, epochs=2, verbose=1)    # , callbacks=[uncertainty])

    if save:
        model.save("probability_general")

    return model, history


class CustomTraining(Sequence):
    def __init__(self, x_train, y_tr, x_val, y_val, model, batch_size):
        self.x_train = x_train
        self.y_train = y_tr[0]
        self.unc = y_tr[1]
        self.x_val = x_val
        self.y_val = y_val[0]
        self.unc_val = y_val[1]
        self.model = model
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x_train) / self.batch_size)

    def __getitem__(self, idx):
        x = self.x_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_b = self.y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        u_b = self.unc[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x, (y_b, u_b)

    def on_epoch_end(self):
        y_p, u_p = self.model.predict(self.x_train)
        y_p = y_p.flatten()
        self.y_train = self.y_train.flatten()
        res = np.absolute(np.subtract(self.y_train, y_p))
        self.unc = res

        mse = mean_squared_error(self.unc, u_p.flatten())
        rmse = np.sqrt(mse)
        RMSE_list.append(rmse)

        mse_y = mean_squared_error(self.y_train, y_p)
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


# Compile and fit general model
def general_model(x_tr, y_tr, save_model=False, load_model=False):

    if load_model:
        loaded_model = tf.keras.models.load_model("general_scoring_model")
        return loaded_model

    history = LossHistory()

    model = Sequential(name="GeneralModel")
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    # for i in range(4):
    #    model.add(Dense(256, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse',
                  optimizer=Nadam(learning_rate=0.001), metrics=MeanAbsoluteError())
    model.fit(x_tr, y_tr, batch_size=64, epochs=1,
              validation_split=0.1, verbose=1, callbacks=[history])

    print(model.summary())

    if save_model:
        model.save("mlp_model")

    return model, history


# Compile and fit a general convolutional neural network model
def cnn_model(x_tr, y_tr, save_model=False):
    history = LossHistory()

    model = Sequential(name="CNN1_model_PP")

    model.add(Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(9, 7, 1)))
    # model.add(MaxPooling2D((4, 4)))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.01), metrics=MeanAbsoluteError())
    print(model.summary())
    model.fit(x_tr, y_tr, batch_size=64, epochs=1, validation_split=0.1,
              verbose=1, callbacks=[history])

    if save_model:
        model.save("cnn_model97")

    return model, history


# Compile and fit a general convolutional neural network model (version 2)
def cnn_model2(x_tr, y_tr, save_model=False):
    history = LossHistory()

    model = Sequential(name="CNN2_model_PP")
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(3, 21, 1)))

    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.005), metrics=MeanAbsoluteError())
    print(model.summary())
    model.fit(x_tr, y_tr, batch_size=32, epochs=1, validation_split=0.1,
              verbose=1, callbacks=[history])

    if save_model:
        model.save("cnn_model321")

    return model, history


# Plot the training history over epochs
def plot_training(hist):
    figure, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(hist.history['loss'], label="training loss")
    ax.plot(hist.history['val_loss'], label="validation loss")
    # ax.plot(hist2.history['loss'], label="CNN_loss")
    ax.set_xlabel("batch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    # ax[0].set_xlim(-0.1, 0.1)
    # ax[0].set_ylim(0.0, 0.1)
    ax.grid()
    ax.legend()

    '''
    ax[1].plot(hist.history['mean_absolute_error'], label="training")
    ax[1].plot(hist.history['val_mean_absolute_error'], label="validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mean absolute error")
    ax[1].set_title("Measure of accuracy")
    # ax[1].set_xlim(-0.1, 0.1)
    # ax[1].set_ylim(0.0, 0.2)
    ax[1].grid()
    ax[1].legend()
    plt.show()
    '''


# Plot the predictions against real values
def plot_predictions(predicted, actual):
    vals = 21
    figure, ax = plt.subplots(1, 1, figsize=(7, 5))
    major_ticks = np.arange(0, 21, 1)

    ax.plot(predicted[0:vals], 'o', label="predicted")
    ax.plot(actual[0:vals], 'x', label="real")
    ax.set_title("Predicted vs real scores of shifts")
    ax.set_xticks(major_ticks)
    ax.grid(which="major", linestyle='--')
    ax.set_xlim(-1, vals)
    ax.set_ylim(-0.01, 1.01)
    ax.legend()

    plt.show()


# Plot the predictions against real values
def plot_predictions2(predicted, actual, vals):
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


# Evaluate model using RMSE and R^2
def model_eval(predicted, actual):
    mse = mean_squared_error(predicted, actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return rmse, r2


# Split data into morning, evening and night shifts
def get_time_shifts(data):
    mornings = []
    evenings = []
    nights = []
    for i in range(len(data)):
        m_shifts = []
        e_shifts = []
        n_shifts = []
        for j in range(0, len(data[i]), 3):     # Go through each day (3 shifts)
            for k in range(3):
                if k == 0:
                    m_shifts.append(data[i, j])
                elif k == 1:
                    e_shifts.append(data[i, j+k])
                else:
                    n_shifts.append(data[i, j+k])
        mornings.append(m_shifts)
        evenings.append(e_shifts)
        nights.append(n_shifts)
    return mornings, evenings, nights


# Display shifts as an image (size=(9,7))
def get_shift_image(data):
    m_shifts, e_shifts, n_shifts = get_time_shifts(data)
    shift = []
    for m, e, n in zip(m_shifts, e_shifts, n_shifts):
        m_sh = np.array(m).reshape(3, 7)
        e_sh = np.array(e).reshape(3, 7)
        n_sh = np.array(n).reshape(3, 7)

        w1 = np.array([m_sh[0], e_sh[0], n_sh[0]])
        w2 = np.array([m_sh[1], e_sh[1], n_sh[1]])
        w3 = np.array([m_sh[2], e_sh[2], n_sh[2]])

        res = np.append(w1, w2, 0)
        res2 = np.append(res, w3, 0)

        shift.append(res2)
    '''
    figure, ax = plt.subplots(2, 2, figsize=(8, 5))

    ax[0, 0].imshow(shift[0], cmap="binary", interpolation='nearest')
    ax[1, 0].imshow(shift[1], cmap="binary", interpolation='nearest')
    ax[0, 1].imshow(shift[2], cmap="binary", interpolation='nearest')
    ax[1, 1].imshow(shift[3], cmap="binary", interpolation='nearest')
    plt.show()
    '''
    return shift


# Display shifts as an image (size=(9,7))
def get_shift_image2(data):
    m_shifts, e_shifts, n_shifts = get_time_shifts(data)
    shift = []
    for m, e, n in zip(m_shifts, e_shifts, n_shifts):
        w1 = np.array([m, e, n])
        shift.append(w1)

    '''
    figure, ax = plt.subplots(2, 2, figsize=(8, 5))

    ax[0, 0].imshow(shift[0], cmap="binary", interpolation='nearest')
    ax[1, 0].imshow(shift[1], cmap="binary", interpolation='nearest')
    ax[0, 1].imshow(shift[2], cmap="binary", interpolation='nearest')
    ax[1, 1].imshow(shift[3], cmap="binary", interpolation='nearest')
    plt.show()
    '''

    return shift


#############################
#       Weigh scores        #
#############################

# Weigh the personal scoring with general scoring to get a final score
def weigh_scores(g_scores, p_scores, w=0.3):
    f_scores = []
    for i in range(len(g_scores)):
        f_scores.append(w * p_scores[i] + (1-w) * g_scores[i])
    return np.array(f_scores)


if __name__ == '__main__':
    d_size = 100000     # Data used for training the model(s)
    add_noise = True    # Adds "noise" to the data by weighing the general score with fictional preferences
    print_train = True     # Prints training history and a sample of predictions
    run_many_test = False

    train_general = False  # Trains the general MLP model
    train_probability = False    # Trains the general probability model
    train_probability2 = False   # Trains the general probability model version2
    train_cnn1 = False    # Trains the general CNN model
    train_cnn2 = True

    per_d = int(0.05*d_size)    # Defines the amount of data to have noise
    weight = 0.2    # Defines the weight that the fictional preferences will have (scale 0-1)
    rmse_mlp, rmse_prob, rmse_prob2, rmse_cnn, rmse_cnn2, \
        r2_mlp, r2_prob, r2_prob2, r2_cnn, r2_cnn2 = np.zeros(10)

    df = pd.read_csv('cleaned_df_new')

    if add_noise:
        n_distr, num_n_shifts = night_distribution(df.iloc[:d_size, :63].values)
        n_scores = weigh_scores(df.iloc[:per_d, 63].values,
                                night_person(n_distr, df.iloc[:per_d, :63].values), w=weight)
        nn_scores = weigh_scores(df.iloc[per_d:2*per_d, 63].values,
                                 non_night_person(n_distr, df.iloc[per_d:2*per_d, :63].values), w=weight)

        w_distr, num_w_shifts = weekend_distribution(df.iloc[:d_size, :63].values)
        w_scores = weigh_scores(df.iloc[2*per_d:3*per_d, 63].values,
                                weekend_person(w_distr, df.iloc[2*per_d:3*per_d, :63].values), w=weight)
        nw_scores = weigh_scores(df.iloc[3*per_d:4*per_d, 63].values,
                                 non_weekend_person(w_distr, df.iloc[3*per_d:4*per_d, :63].values), w=weight)

        df.iloc[:per_d, 63] = n_scores
        df.iloc[per_d:2*per_d, 63] = nn_scores
        df.iloc[2*per_d:3*per_d, 63] = w_scores
        df.iloc[3*per_d:4*per_d, 63] = nw_scores

    # Split into X and y values
    X = df.iloc[:d_size, :63].values
    y = df.iloc[:d_size, 63].values

    # Create training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    if run_many_test:
        loss_hist = []
        bad_loss_hist = []
        rmse_cnn_array = []
        r2_cnn_array = []
        count_bad_runs = 0
        count_total_runs = 0
        count_good_runs = 0
        while count_good_runs < 50:
            # Create training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

            # Create a graphical shift format
            g_train_X = get_shift_image2(X_train)
            g_test_X = get_shift_image2(X_test)

            g_train_X = np.array(g_train_X).reshape(-1, 3, 21, 1)
            g_test_X = np.array(g_test_X).reshape(-1, 3, 21, 1)

            # Train CNN general model and predict unseen data
            cnn_mod2, cnn_history2 = cnn_model2(g_train_X, y_train)
            y_pred_cnn2 = cnn_mod2.predict(g_test_X)
            rmse_cnn2, r2_cnn2 = model_eval(y_pred_cnn2, y_test)
            count_total_runs += 1

            if r2_cnn2 > 0:
                rmse_cnn_array.append(rmse_cnn2)
                r2_cnn_array.append(r2_cnn2)
                loss_hist.append(cnn_history2.history['loss'])
            else:
                bad_loss_hist.append(cnn_history2.history['loss'])
                count_bad_runs += 1

            count_good_runs = count_total_runs - count_bad_runs

        print("Mean of RMSE: ", mean(rmse_cnn_array))
        print("Mean of R2: ", mean(r2_cnn_array))
        print("Bad runs:", count_bad_runs, ", Proportion of bad runs: ", count_bad_runs/count_total_runs)

        df_rmse = pd.DataFrame(rmse_cnn_array, columns=['RMSE score'])
        df_r2 = pd.DataFrame(r2_cnn_array, columns=['R2 score'])

        # figure, ax = plt.subplots(1, 2, figsize=(10, 5))

        sns.boxplot(y='RMSE score', data=df_rmse, width=0.4)
        sns.stripplot(y='RMSE score', color='black', data=df_rmse)
        plt.show()

        sns.boxplot(y='R2 score', data=df_r2, width=0.4)
        sns.stripplot(y='R2 score', color='black', data=df_r2)
        plt.show()

        fig, axis = plt.subplots(1, 2, figsize=(13, 5))

        axis[0].boxplot(rmse_cnn_array, labels=['RMSE score'])
        axis[1].boxplot(r2_cnn_array, labels=['R2 score'])
        plt.show()

        for loss in loss_hist:
            plt.plot(loss)
        plt.title("Batch loss for 50 runs with MLP model")
        plt.grid()
        plt.show()

    if train_general:
        # Train general model and predict unseen data
        mlp_model, mlp_history = general_model(X_train, y_train, save_model=False)
        y_pred_mlp = mlp_model.predict(X_test)
        rmse_mlp, r2_mlp = model_eval(y_pred_mlp, y_test)

        if print_train:
            plot_training(mlp_history)
            plot_predictions(y_pred_mlp, y_test)

    if train_probability:
        # Train general probability model and predict unseen data
        prob_model, prob_history = general_probability_model(X_train, y_train)
        y_pred_prob = prob_model.predict(X_test)
        rmse_prob, r2_prob = model_eval(y_pred_prob, y_test)

        if print_train:
            plot_training(prob_history)
            plot_predictions(y_pred_prob, y_test)

    if train_probability2:
        bins = 200

        # Train general probability model and predict unseen data
        prob_model2, prob_history2 = general_probability(X_train, y_train, save=False)
        y_pred_prob2 = prob_model2.predict(X_test)
        rmse_prob2, r2_prob2 = model_eval(y_pred_prob2[:][0], y_test)

        pred_error = y_pred_prob2[:][1]
        pred_score = y_pred_prob2[:][0]
        actual_error = np.absolute(pred_score.flatten() - y_test)

        ind = np.arange(len(pred_error))
        dataframe2 = pd.DataFrame({'Index': ind.flatten(),
                                   'Pred_error': pred_error.flatten(),
                                   'Actual_error': actual_error.flatten()})
        df_p_error = dataframe2.sort_values(by=['Pred_error'])
        df_a_error = dataframe2.sort_values(by=['Actual_error'])
        print(df_p_error, len(df_p_error))
        print(df_a_error, len(df_a_error))

        avg_pred = sum(pred_error) / len(pred_error)
        avg_actual = sum(actual_error) / len(actual_error)

        nr_values = int(len(df_a_error)/bins)
        bin_means_a = []
        bin_means_p = []
        for val in range(0, len(df_a_error), nr_values):
            vals_bin_a = df_a_error['Actual_error'].iloc[val:val + nr_values]
            vals_bin_p = df_a_error['Pred_error'].iloc[val:val + nr_values]
            bin_means_a.append(mean(vals_bin_a))
            bin_means_p.append(mean(vals_bin_p))

        plt.figure(1)
        plt.plot(bin_means_a, label='Actual errors')
        plt.legend()
        plt.grid()

        plt.figure(2)
        plt.plot(bin_means_p, label='Predicted errors')
        plt.legend()
        plt.grid()
        plt.show()

        '''
        n_worst = 100
        
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

        print(int(0.9 * len(y_train)), len(y_test))
        print("Average of predicted errors: ", avg_pred)
        print("Average of actual errors: ", avg_actual)
        print("Average of", n_worst, "worst predicted errors (ordered by predicted worsts): ", avg_pred_worst)
        print("Average of", n_worst, "worst actual errors (ordered by actual worsts): ", avg_actual_worst)
        print("Average of", n_worst, "worst predicted errors (ordered by actual worsts): ", avg_pred_worst_r)
        print("Average of", n_worst, "worst actual errors (ordered by predicted worsts): ", avg_actual_worst_r)
        '''
        if print_train:
            # plot_training(prob_history2)
            plot_predictions2(y_pred_prob2, y_test, vals=21)
            plt.plot(RMSE_list)
            plt.plot(RMSE_list_val)
            plt.show()

    if train_cnn1:
        # Create a graphical shift format
        g_train_X = get_shift_image(X_train)
        g_test_X = get_shift_image(X_test)

        g_train_X = np.array(g_train_X).reshape(-1, 9, 7, 1)
        g_test_X = np.array(g_test_X).reshape(-1, 9, 7, 1)

        # Train CNN general model and predict unseen data
        cnn_model, cnn_history = cnn_model(g_train_X, y_train, save_model=False)
        y_pred_cnn = cnn_model.predict(g_test_X)
        rmse_cnn, r2_cnn = model_eval(y_pred_cnn, y_test)

        if print_train:
            plot_training(cnn_history)
            plot_predictions(y_pred_cnn, y_test)

    if train_cnn2:
        rmse_cnn_arr = []
        r2_cnn_arr = []
        models = []
        good_runs = 0
        runs = 0

        while good_runs < 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

            # Create a graphical shift format
            g_train_X = get_shift_image2(X_train)
            g_test_X = get_shift_image2(X_test)

            g_train_X = np.array(g_train_X).reshape(-1, 3, 21, 1)
            g_test_X = np.array(g_test_X).reshape(-1, 3, 21, 1)

            # Train CNN general model and predict unseen data
            cnn_mod2, cnn_history2 = cnn_model2(g_train_X, y_train, save_model=False)
            y_pred_cnn2 = cnn_mod2.predict(g_test_X)
            rmse_cnn2, r2_cnn2 = model_eval(y_pred_cnn2, y_test)
            runs += 1

            if r2_cnn2 > -1:
                rmse_cnn_arr.append(rmse_cnn2)
                r2_cnn_arr.append(r2_cnn2)
                models.append(cnn_mod2)
                good_runs += 1

        df_rmse = pd.DataFrame(np.array(rmse_cnn_arr).T, columns=['RMSE'])
        df_rmse = df_rmse.melt(var_name='CNN2', value_name='RMSE score')

        df_r2 = pd.DataFrame(np.array(r2_cnn_arr).T, columns=['R2'])
        df_r2 = df_r2.melt(var_name='CNN2', value_name='R2 score')

        sns.boxplot(x='CNN2', y='RMSE score', data=df_rmse, width=0.5)
        sns.stripplot(x='CNN2', y='RMSE score', color='black', data=df_rmse)
        plt.grid(axis='both')
        plt.show()

        sns.boxplot(x='CNN2', y='R2 score', data=df_r2, width=0.5)
        sns.stripplot(x='CNN2', y='R2 score', color='black', data=df_r2)
        plt.grid(axis='both')
        plt.show()

        print('Percentage of good runs: ', good_runs/runs)
        print('RMSE mean: ', mean(rmse_cnn_arr))
        print('R2 mean: ', mean(r2_cnn_arr))

        min_rmse = min(rmse_cnn_arr)
        ind = rmse_cnn_arr.index(min_rmse)
        mod_save = models[ind]
        # mod_save.save("cnn_model321_ver2")

        print('RMSE list: ', rmse_cnn_arr)
        print('Min RMSE: ', min_rmse)
        print('Index: ', ind)

        if print_train:
            plot_training(cnn_history2)
            plot_predictions(y_pred_cnn2, y_test)

    print("RMSE (mlp): ", rmse_mlp, "R2 (mlp): ", r2_mlp)
    print("RMSE (prob): ", rmse_prob, "R2 (prob): ", r2_prob)
    print("RMSE (prob2): ", rmse_prob2, "R2 (prob2): ", r2_prob2)
    print("RMSE (cnn 9x7): ", rmse_cnn, "R2 (cnn 9x7): ", r2_cnn)
    print("RMSE (cnn 3x21): ", rmse_cnn2, "R2 (cnn 3x21): ", r2_cnn2)

    '''
    res_rmse = np.array([0.0864, 0.0793, 0.0607, 0.0552, 0.0537, 0.0507, 0.0465, 0.0436, 0.0421])
    res_r2 = np.array([0.7257, 0.7642, 0.8377, 0.8644, 0.8704, 0.8851, 0.9035, 0.9153, 0.9215])
    train_data = np.array([100000, 200000, 500000, 800000, 1000000, 1200000, 1500000, 1800000, 2000000])

    plt.plot(train_data, res_rmse)
    plt.xlabel("Data used")
    plt.ylabel("RMSE score")
    plt.show()

    plt.plot(train_data, res_r2)
    plt.xlabel("Data used")
    plt.ylabel("R2 score")
    plt.show()
    '''
