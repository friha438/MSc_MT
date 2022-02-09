import pandas as pd
import numpy as np
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsoluteError, Hinge
from tensorflow.keras.optimizers import Adam, SGD
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
def night_person(dist, data):
    n_nights = count_nights(data)
    static_score = dist.cdf(n_nights)
    score = np.random.normal(loc=static_score, scale=0.05)
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
    static_score = abs(np.array(static_score_night) - 1)
    score = np.random.normal(loc=static_score, scale=0.05)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


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


def weekend_distribution(data):
    weekend_shifts = count_weekends(data)
    weekend_hist = np.histogram(weekend_shifts, bins=np.arange(11))
    weekend_dist = rv_histogram(weekend_hist)
    return weekend_dist, weekend_shifts


# Create a person that will score high if there are many weekend shifts
def weekend_person(dist, data):
    weekend_shifts = count_weekends(data)

    static_score = dist.cdf(weekend_shifts)
    score = np.random.normal(loc=static_score, scale=0.05)
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
    static_score = abs(np.array(static_score_w) - 1)
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
    model.add(Dense(128, activation='relu'))
    # for i in range(4):
    #    model.add(Dense(256, activation='tanh'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=MeanAbsoluteError())  # Add accuracy
    history = model.fit(x_train, y_train, batch_size=64, epochs=150, validation_data=(x_val, y_val), verbose=0)

    print(model.summary())

    return model, history


# Plot the training history over epochs
def plot_training(hist1, hist2):
    figure, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].plot(hist1.history['loss'], label="training")
    ax[0, 0].plot(hist1.history['val_loss'], label="validation")
    ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("loss")
    ax[0, 0].set_title("Training on night person data")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(hist1.history['mean_absolute_error'], label="training")
    ax[1, 0].plot(hist1.history['val_mean_absolute_error'], label="validation")
    ax[1, 0].set_xlabel("epoch")
    ax[1, 0].set_ylabel("mean absolute error")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(hist2.history['loss'], label="training")
    ax[0, 1].plot(hist2.history['val_loss'], label="validation")
    ax[0, 1].set_xlabel("epoch")
    ax[0, 1].set_ylabel("loss")
    ax[0, 1].set_title("Training on weekend person data")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(hist2.history['mean_absolute_error'], label="training")
    ax[1, 1].plot(hist2.history['val_mean_absolute_error'], label="validation")
    ax[1, 1].set_xlabel("epoch")
    ax[1, 1].set_ylabel("mean absolute error")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()


# Plot the predictions against real values
def plot_predictions(pred1, pred2, test1, test2):
    vals = 21
    figure, ax = plt.subplots(1, 2, figsize=(14, 5))
    major_ticks = np.arange(0, 21, 1)

    ax[0].plot(pred1[0:vals], 'o', label="predicted")
    ax[0].plot(test1[0:vals], 'x', label="real")
    ax[0].set_title("Predicted vs real night data")
    ax[0].set_xticks(major_ticks)
    ax[0].grid(which="major", linestyle='--')
    ax[0].set_xlim(-1, vals)
    ax[0].set_ylim(-0.01, 1.01)
    ax[0].legend()

    ax[1].plot(pred2[0:vals], 'o', label="predicted")
    ax[1].plot(test2[0:vals], 'x', label="real")
    ax[1].set_title("Predicted vs real weekend data")
    ax[1].set_xticks(major_ticks)
    ax[1].grid(which="major", linestyle='--')
    ax[1].set_xlim(-1, vals)
    ax[1].set_ylim(-0.01, 1.01)
    ax[1].legend()

    plt.show()


# Evaluate model using RMSE and R^2
def model_eval(predicted, actual):
    mse = mean_squared_error(predicted, actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return rmse, r2


if __name__ == '__main__':

    startTime = datetime.now()  # For measuring execution time
    d_size = 1000000  # How many shifts are used
    q_answers = 500000  # How many scored rosters there are
    q_val = 10000      # How many scored rosters used for validation
    get_preferences = True
    PP_model = False    # If set to true, then night_p also needs to be true
    PP_model_iter = True
    other = False

    # Read data
    dataframe = read_data_df(d_size)

    # split data for training, validation, and testing
    train_set, test_set = train_test_split(dataframe.values, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.1)

    morning_shifts, evening_shifts, night_shifts = get_time_shifts(train_set)
    morning_shifts_val, evening_shifts_val, night_shifts_val = get_time_shifts(val_set)
    morning_shifts_test, evening_shifts_test, night_shifts_test = get_time_shifts(test_set)

    if get_preferences:
        # Score data based on defined preferences regarding night shifts
        n_distr = night_distribution(night_shifts)
        n_scores = night_person(n_distr, night_shifts[0:q_answers])
        d_scores = non_night_person(n_distr, night_shifts[0:q_answers])
        n_scores_val = night_person(n_distr, night_shifts_val[0:q_val])
        d_scores_val = non_night_person(n_distr, night_shifts_val[0:q_val])
        n_scores_test = night_person(n_distr, night_shifts_test)
        d_scores_test = non_night_person(n_distr, night_shifts_test)

        # Score data based on defined preferences regarding weekend shifts
        w_dist, w_shifts = weekend_distribution(train_set)
        w_scores = weekend_person(w_dist, train_set[0:q_answers])
        nw_scores = non_weekend_person(w_dist, train_set[0:q_answers])
        w_scores_val = weekend_person(w_dist, val_set[0:q_val])
        nw_scores_val = non_weekend_person(w_dist, val_set[0:q_val])
        w_scores_test = weekend_person(w_dist, test_set)
        nw_scores_test = non_weekend_person(w_dist, test_set)

    if PP_model:
        # Create and train model
        pp_model_n, pp_hist_n = personal_preference_model(train_set[0:q_answers], val_set[0:q_val],
                                                          n_scores, n_scores_val)
        pp_model_w, pp_hist_w = personal_preference_model(train_set[0:q_answers], val_set[0:q_val],
                                                          w_scores, w_scores_val)

        # Predict and evaluate
        pred_val_n = pp_model_n.predict(train_set[0:q_answers])
        pred_val_w = pp_model_w.predict(train_set[0:q_answers])

        rmse_score_n, r2_score_n = model_eval(pred_val_n, n_scores)
        rmse_score_w, r2_score_w = model_eval(pred_val_w, w_scores)
        rmse_random, r2_random = model_eval(w_scores_test[0:1000], n_scores_test[0:1000])

        print("Execution time: ", datetime.now() - startTime)
        print("RMSE (night person): ", rmse_score_n)
        print("RMSE (weekend person): ", rmse_score_w)
        print("R2 (night person): ", r2_score_n)
        print("R2 (weekend person): ", r2_score_w)
        print("Random RMSE and R2: ", rmse_random, r2_random)

        plot_training(pp_hist_n, pp_hist_w)
        plot_predictions(pred_val_n, pred_val_w, n_scores, w_scores)

    if PP_model_iter:
        data_used = np.arange(100, 5000, 100)
        rmse_w = []
        rmse_n = []
        r2_w = []
        r2_n = []

        for d in data_used:
            # Create and train model
            pp_model_n, pp_hist_n = personal_preference_model(train_set[0:d], val_set[0:d],
                                                              n_scores[0:d], n_scores_val[0:d])
            pp_model_w, pp_hist_w = personal_preference_model(train_set[0:d], val_set[0:d],
                                                              w_scores[0:d], w_scores_val[0:d])

            # Predict and evaluate
            pred_val_n = pp_model_n.predict(train_set[0:d])
            pred_val_w = pp_model_w.predict(train_set[0:d])

            rmse_score_n, r2_score_n = model_eval(pred_val_n, n_scores[0:d])
            rmse_score_w, r2_score_w = model_eval(pred_val_w, w_scores[0:d])

            rmse_n.append(rmse_score_n)
            rmse_w.append(rmse_score_w)
            r2_n.append(r2_score_n)
            r2_w.append(r2_score_w)

        figure, ax = plt.subplots(2, 2, figsize=(12, 12))

        ax[0, 0].plot(data_used, rmse_n, label="RMSE night")
        ax[0, 0].set_xlabel("data used for training")
        ax[0, 0].set_ylabel("RMSE")
        ax[0, 0].grid()
        ax[0, 0].legend()

        ax[1, 0].plot(data_used, r2_n, label="R2-score night")
        ax[1, 0].set_xlabel("data used for training")
        ax[1, 0].set_ylabel("R2 score")
        ax[1, 0].grid()
        ax[1, 0].legend()

        ax[0, 1].plot(data_used, rmse_w, label="RMSE weekend")
        ax[0, 1].set_xlabel("data used for training")
        ax[0, 1].set_ylabel("RMSE")
        ax[0, 1].grid()
        ax[0, 1].legend()

        ax[1, 1].plot(data_used, r2_n, label="R2-score night")
        ax[1, 1].set_xlabel("data used for training")
        ax[1, 1].set_ylabel("R2 score")
        ax[1, 1].grid()
        ax[1, 1].legend()

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
