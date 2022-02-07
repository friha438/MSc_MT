import pandas as pd
import numpy as np
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime


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


# Split data into morning, evening and night shifts
def get_time_shifts(data):
    mornings = []
    evenings = []
    nights = []
    for i in range(len(data)):
        m_shifts = []
        e_shifts = []
        n_shifts = []
        for j in range(0, len(data[i]), 3):
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


# Count how many night shifts each person has
def count_nights(data):
    n_nights = []
    for i in range(len(data)):
        n_nights.append(np.sum(data[i]))

    return n_nights


# Create a distribution of night shifts
def night_distribution(data):
    n_nights = count_nights(data)
    night_hist = np.histogram(n_nights, bins=np.arange(11))
    night_dist = rv_histogram(night_hist)

    return night_dist


# Create a person to score good on many nights and bad on few
def night_person(distr, data):
    n_nights = count_nights(data)
    static_score = distr.cdf(n_nights)
    score = np.random.normal(loc=static_score, scale=0.05)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


# Create a person to score good on few nights and bad on many
def non_night_person(distr, data):
    n_nights = count_nights(data)
    static_score_night = distr.cdf(n_nights)
    static_score = abs(np.array(static_score_night) - 1)
    score = np.random.normal(loc=static_score, scale=0.05)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


def weekend_distribution(data):
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

    weekend_hist = np.histogram(weekend_shifts, bins=np.arange(11))
    weekend_dist = rv_histogram(weekend_hist)
    return weekend_dist, weekend_shifts


def weekend_person(data):
    weekend_dist, weekend_shifts = weekend_distribution(data)

    static_score = weekend_dist.cdf(weekend_shifts)
    score = np.random.normal(loc=static_score, scale=0.05)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0

    return score


# Compile and fit pp model
def personal_preference_model(x_train, x_val, y_train, y_val):
    model = Sequential(name="PersonalPreferenceModel")
    model.add(Dense(256, activation='relu'))
    # for i in range(4):
    #    model.add(Dense(256, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val), verbose=1)

    print(model.summary())

    return model, history


# Plot the training history over epochs
def plot_training(hist):
    plt.plot(hist.history['loss'], label="training")
    plt.plot(hist.history['val_loss'], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend()
    plt.show()


# Evaluate model using RMSE
def model_eval(predicted, actual):
    mse = mean_squared_error(predicted, actual)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == '__main__':
    startTime = datetime.now()  # For measuring execution time
    d_size = 100000  # How many shifts are used
    q_answers = 500  # How many scored rosters there are
    q_val = 10      # How many scored rosters used for validation
    night_p = True
    PP_model = False    # If set to true, then night_p also needs to be true
    other = False

    # Read data
    dataframe = read_data_df(d_size)

    # split data for training, validation, and testing
    train_set, test_set = train_test_split(dataframe.values, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.1)

    morning_shifts, evening_shifts, night_shifts = get_time_shifts(train_set)
    morning_shifts_val, evening_shifts_val, night_shifts_val = get_time_shifts(val_set)
    morning_shifts_test, evening_shifts_test, night_shifts_test = get_time_shifts(test_set)

    # Score data based on defined preferences
    n_distr = night_distribution(night_shifts)
    n_scores = night_person(n_distr, night_shifts[0:q_answers])
    d_scores = non_night_person(n_distr, night_shifts[0:q_answers])
    n_scores_val = night_person(n_distr, night_shifts_val[0:q_val])
    d_scores_val = non_night_person(n_distr, night_shifts_val[0:q_val])

    w_dist, w_shifts = weekend_distribution(train_set[0:1000])
    w_scores = weekend_person(train_set[0:100])
    print("mornings: ", morning_shifts[0:3])
    print("evenings: ", evening_shifts[0:3])
    print("nights: ", night_shifts[0:3])

    plt.plot(w_shifts[0:100], w_scores, 'o')
    plt.show()

    if PP_model:
        # Create and train model
        pp_model, pp_hist = personal_preference_model(train_set[0:q_answers], val_set[0:q_val], n_scores, n_scores_val)

        # Predict and evaluate
        pred_val = pp_model.predict(test_set)
        n_score_test = night_person(n_distr, night_shifts_test)
        rmse_score = model_eval(pred_val, n_score_test)
        rmse_random = model_eval(n_score_test[0:q_answers], n_scores)

        print("Execution time: ", datetime.now() - startTime)
        print("RMSE: ", rmse_score)
        print("SI: ", rmse_score/np.mean(n_scores))
        print("RMSE random: ", rmse_random)

        plot_training(pp_hist)

        plt.plot(pred_val[0:30], 'o', label="predicted")
        plt.plot(n_score_test[0:30], 'x', label="real")
        plt.grid()
        plt.legend()
        plt.show()

    if other:
        '''
        q_ans = np.arange(20, q_answers, 20)
        rmse_score = []
        rmse_random = []
        for q in q_ans:
            # Create and train model
            pp_model, pp_hist = personal_preference_model(train_set[0:q], val_set[0:q_val], n_scores[0:q], n_scores_val)
    
            # Predict and evaluate
            pred_val = pp_model.predict(test_set)
            n_score_test, night_shifts = night_person(n_distr, test_set)
            rmse_score.append(model_eval(pred_val, n_score_test))
            rmse_random.append(model_eval(n_score_test[0:q_answers], n_scores))
    
        plt.plot(q_ans, rmse_score, label="RMSE")
        plt.xlabel("Data used for training")
        plt.ylabel("RMSE")
        plt.grid()
        plt.legend()
        plt.show()
    
        plt.plot(q_ans, rmse_random, label="RMSE random")
        plt.xlabel("Data used for training")
        plt.ylabel("RMSE random")
        plt.grid()
        plt.legend()
        plt.show()
        '''

        # plt.plot(n_night_shifts, n_scores, 'o')
        # plt.plot(n_night_shifts, d_scores, 'x')
        # plt.show()
        # plot_training(pp_hist)

        '''
        fig, ax = plt.subplots()
        ax.grid(axis='y')
        plt.hist(night_shifts, bins=np.arange(12)-0.5, rwidth=0.8)
        plt.xticks(range(12))
        plt.xlabel("Number of night shifts in a roster per person")
        plt.ylabel("Distribution of night shifts in rosters")
        plt.show()
        '''
