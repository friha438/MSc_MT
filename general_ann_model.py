import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_histogram
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from datetime import datetime


#############################
#       Read data           #
#############################

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


# Read scores given for each shift
def read_scores(size):
    a = []
    f = open("scores.txt", "r")
    for i in range(size):
        r = float(f.readline())
        a.append(r)
    return a


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
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))


# Compile and fit general model
def general_model(x_tr, x_val, y_tr, y_val, save_model=False, load_model=False):

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
              validation_data=(x_val, y_val), verbose=1, callbacks=[history])

    print(model.summary())

    if save_model:
        model.save("g_scoring_model_new")

    return model, history


# Plot the training history over epochs
def plot_training(hist):
    figure, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(hist.history['loss'], label="training")
    ax.plot(hist.history['val_loss'], label="validation")
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


# Evaluate model using RMSE and R^2
def model_eval(predicted, actual):
    mse = mean_squared_error(predicted, actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return rmse, r2


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
    startTime = datetime.now()  # For measuring execution time
    # d_size = 3307321   # How much data to use
    d_size = 100000
    get_pers_scores = False
    create_sets = False
    per_d = int(0.05*d_size)
    per_t = int(0.1*per_d)
    w = 0.2

    # Read data
    shift_data = read_data_df(d_size).values
    shift_scores = np.array(read_scores(d_size))

    # Scale the scores
    scaler = MinMaxScaler()
    sc_scores = scaler.fit_transform(shift_scores.reshape(-1, 1))
    
    if get_pers_scores:
        n_distr, n_shifts = night_distribution(shift_data[:10000])
        n_scores = weigh_scores(sc_scores[:per_d],
                                night_person(n_distr, shift_data[:per_d]), w=w)
        n_scores_val = weigh_scores(sc_scores[per_d:per_d+per_t],
                                    night_person(n_distr, shift_data[per_d:per_d+per_t]), w=w)
        n_scores_test = weigh_scores(sc_scores[per_d+per_t:per_d+2*per_t],
                                     night_person(n_distr, shift_data[per_d+per_t:per_d+2*per_t]), w=w)

        nn_scores = weigh_scores(sc_scores[per_d+2*per_t:2*per_d+2*per_t],
                                 non_night_person(n_distr, shift_data[per_d+2*per_t:2*per_d+2*per_t]), w=w)
        nn_scores_val = weigh_scores(sc_scores[2*per_d+2*per_t:2*per_d+3*per_t],
                                     non_night_person(n_distr, shift_data[2*per_d+2*per_t:2*per_d+3*per_t]), w=w)
        nn_scores_test = weigh_scores(sc_scores[2*per_d+3*per_t:2*per_d+4*per_t],
                                      non_night_person(n_distr, shift_data[2*per_d+3*per_t:2*per_d+4*per_t]), w=w)

        w_distr, w_shifts = weekend_distribution(shift_data[:10000])
        w_scores = weigh_scores(sc_scores[2*per_d+4*per_t:3*per_d+4*per_t],
                                weekend_person(w_distr, shift_data[2*per_d+4*per_t:3*per_d+4*per_t]), w=w)
        w_scores_val = weigh_scores(sc_scores[3*per_d+4*per_t:3*per_d+5*per_t],
                                    weekend_person(w_distr, shift_data[3*per_d+4*per_t:3*per_d+5*per_t]), w=w)
        w_scores_test = weigh_scores(sc_scores[3*per_d+5*per_t:3*per_d+6*per_t],
                                     weekend_person(w_distr, shift_data[3*per_d+5*per_t:3*per_d+6*per_t]), w=w)

        nw_scores = weigh_scores(sc_scores[3*per_d+6*per_t:4*per_d+6*per_t],
                                 non_weekend_person(w_distr, shift_data[3*per_d+6*per_t:4*per_d+6*per_t]), w=w)
        nw_scores_val = weigh_scores(sc_scores[4*per_d+6*per_t:4*per_d+7*per_t],
                                     non_weekend_person(w_distr, shift_data[4*per_d+6*per_t:4*per_d+7*per_t]), w=w)
        nw_scores_test = weigh_scores(sc_scores[4*per_d+7*per_t:4*per_d+8*per_t],
                                      non_weekend_person(w_distr, shift_data[4*per_d+7*per_t:4*per_d+8*per_t]), w=w)

    if create_sets:
        sc_scores[:per_d] = n_scores
        sc_scores[per_d:per_d+per_t] = n_scores_val
        sc_scores[per_d+per_t:per_d+2*per_t] = n_scores_test

        sc_scores[per_d+2*per_t:2*per_d+2*per_t] = nn_scores
        sc_scores[2*per_d+2*per_t:2*per_d+3*per_t] = nn_scores_val
        sc_scores[2*per_d+3*per_t:2*per_d+4*per_t] = nn_scores_test

        sc_scores[2*per_d+4*per_t:3*per_d+4*per_t] = w_scores
        sc_scores[3*per_d+4*per_t:3*per_d+5*per_t] = w_scores_val
        sc_scores[3*per_d+5*per_t:3*per_d+6*per_t] = w_scores_test

        sc_scores[3*per_d+6*per_t:4*per_d+6*per_t] = nw_scores
        sc_scores[4*per_d+6*per_t:4*per_d+7*per_t] = nw_scores_val
        sc_scores[4*per_d+7*per_t:4*per_d+8*per_t] = nw_scores_test

    X_train, X_test, y_train, y_test = train_test_split(shift_data, sc_scores, test_size=0.2, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    # Create and train model
    g_model, g_history = general_model(X_train, X_valid, y_train, y_valid, save_model=True)
    y_pred = g_model.predict(X_test)

    # Evaluate model
    rmse_sc, r2_sc = model_eval(y_pred, y_test)
    print("RMSE: ", rmse_sc)
    print("R2: ", r2_sc)
    print("Execution time: ", datetime.now() - startTime)

    plot_training(g_history)
    plot_predictions(y_pred, y_test)

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
