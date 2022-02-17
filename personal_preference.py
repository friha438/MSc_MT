import pandas as pd
import numpy as np
from scipy.stats import rv_histogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import SGD
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


##########################################
#      Split shifts based on timing      #
##########################################

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


#####################################
#      Night shift preferences      #
#####################################

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


###############################################
#      Combine pref with general scoring      #
###############################################

# Read scores given for each shift
def read_scores(size):
    a = []
    f = open("scores.txt", "r")
    for i in range(size):
        r = float(f.readline())
        a.append(r)
    return a


# Weigh the personal scoring with general scoring to get a final score
def weigh_scores(g_scores, p_scores, w=0.3):
    f_scores = []
    for i in range(len(g_scores)):
        f_scores.append(w * p_scores[i] + (1-w) * g_scores[i])
    return np.array(f_scores)


########################################
#      Train and evaluate network      #
########################################

# Compile and fit pp model
def personal_preference_model(x_train, x_val, y_train, y_val):
    loaded_model = tf.keras.models.load_model("g_scoring_model_new")

    # Freeze some layers
    for layer in loaded_model.layers[:3]:
        layer.trainable = False

    # Check the trainable status of layers
    for layer in loaded_model.layers:
        print(layer, layer.trainable)

    model = Sequential(name="PersonalPreferenceModel")
    # for layer in loaded_model.layers[0:3]:
    # model.add(layer)

    model.add(loaded_model)
    # model.add(Dense(512, activation='relu', name="new_layer"))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='sigmoid', name="fine_tune"))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_val, y_val), verbose=0)

    print(model.summary())

    return model, history


# Plot the training history over epochs
def plot_training(hist1):
    figure, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(hist1.history['loss'], label="training")
    ax[0].plot(hist1.history['val_loss'], label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Training on non-weekend person (loss)")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(hist1.history['mean_absolute_error'], label="training")
    ax[1].plot(hist1.history['val_mean_absolute_error'], label="validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mean absolute error")
    ax[1].set_title("Training on non-weekend person (MAE)")
    ax[1].grid()
    ax[1].legend()

    plt.show()

    '''
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
    '''


# Plot the predictions against real values
def plot_predictions(pred1, pred2, test1, test2):
    vals = 16
    figure, ax = plt.subplots(1, 2, figsize=(14, 5))
    major_ticks = np.arange(0, 16, 1)

    ax[0].plot(pred1[0:vals], 'o', label="predicted")
    ax[0].plot(test1[0:vals], 'x', label="real")
    ax[0].set_title("Predicted vs real weekend weighted scores")
    ax[0].set_xticks(major_ticks)
    ax[0].grid(which="major", linestyle='--')
    ax[0].set_xlim(-1, vals)
    ax[0].set_ylim(-0.01, 1.01)
    ax[0].legend()

    ax[1].plot(pred2[0:vals], 'o', label="predicted")
    ax[1].plot(test2[0:vals], 'x', label="real")
    ax[1].set_title("Predicted vs real unweighted data")
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
    d_size = 100000  # How many shifts are used
    d_s = 3307321    # Read all scores (To scale them right)
    q_answers = 10000  # How many scored rosters there are
    test_d = 1000    # How many data points used for testing
    weight = 0.4
    get_preferences = True
    PP_model = False    # If set to true, then night_p also needs to be true
    PP_model_iter = True
    other = False

    # Read data
    dataframe = read_data_df(d_size)
    gen_scores = np.array(read_scores(d_s))

    # Scale the scores
    scaler = MinMaxScaler()
    sc_scores = scaler.fit_transform(gen_scores.reshape(-1, 1))
    scores_f = sc_scores[:d_size]

    # split data for training and testing
    train_X, test_X, train_y, test_y = train_test_split(dataframe.values, scores_f, test_size=0.2, shuffle=False)

    # Load the pre-trained model (to compare against fine-tuned model)
    loaded_model2 = tf.keras.models.load_model("g_scoring_model_new")
    preds = loaded_model2.predict(train_X)

    if get_preferences:
        # Score data based on defined preferences regarding weekend shifts
        w_dist, w_shifts = weekend_distribution(train_X)
        nw_scores = non_weekend_person(w_dist, train_X[:q_answers])     # Get raw scores (q_answers)
        nw_scores, nw_scores_val = train_test_split(nw_scores, test_size=0.2, shuffle=False)
        nw_scores_test = non_weekend_person(w_dist, test_X[:test_d])     # Get raw test scores

        morning_shifts, evening_shifts, night_shifts = get_time_shifts(train_X)
        morning_shifts_test, evening_shifts_test, night_shifts_test = get_time_shifts(test_X)
        n_distr = night_distribution(night_shifts)
        n_scores = non_night_person(n_distr, night_shifts[:q_answers])
        n_scores, n_scores_val = train_test_split(n_scores, test_size=0.2, shuffle=False)
        n_scores_test = non_night_person(n_distr, test_X)

        '''
        weighted = weigh_scores(n_scores, nw_scores, w=1)
        weighted_test = weigh_scores(n_scores_test, nw_scores_test, w=1)
        n_nights = count_nights(night_shifts)

        figure, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(w_shifts[:len(weighted)], weighted, 'o')
        ax[1].plot(n_nights[:len(weighted)], weighted, 'x')
        plt.show()
        '''

        nw_weighted = weigh_scores(train_y[:len(nw_scores)], nw_scores, w=weight)
        nw_weighted_val = weigh_scores(train_y[len(nw_scores):q_answers], nw_scores_val, w=weight)
        nw_weighted_test = weigh_scores(test_y[:test_d], nw_scores_test, w=weight)

    if PP_model:

        # Create and train model
        pp_model_w, pp_hist_w = personal_preference_model(train_X[:len(nw_scores)],             # Training data
                                                          train_X[len(nw_scores):q_answers],    # Validation data
                                                          # train_y[:len(nw_scores)],
                                                          # train_y[len(nw_scores):q_answers])
                                                          nw_weighted, nw_weighted_val)         # Train and val data

        # Predict and evaluate
        pred_val_w = pp_model_w.predict(test_X[:test_d])
        rmse_score_w, r2_score_w = model_eval(pred_val_w, nw_weighted_test[:test_d])

        print("Execution time: ", datetime.now() - startTime)
        print("RMSE (weekend person): ", rmse_score_w)
        print("R2 (weekend person): ", r2_score_w)

        plt.plot(w_shifts[0:len(nw_weighted)], nw_weighted, 'o')
        plt.show()

        preds2 = loaded_model2.predict(test_X[:len(nw_scores)])
        plot_training(pp_hist_w)
        plot_predictions(pred_val_w, preds2, nw_weighted_test[:len(nw_scores)], test_y[:len(nw_scores)])

    if PP_model_iter:

        # w = 0.5
        weights = np.arange(0.0, 1.1, 0.1)
        d = 1000
        rmse_w = []
        r2_w = []

        for w in weights:
            nw_weighted = weigh_scores(train_y[:len(nw_scores)], nw_scores, w=w)
            nw_weighted_test = weigh_scores(test_y[:test_d], nw_scores_test, w=w)

            # Create and train model
            pp_model_w, pp_hist_w = personal_preference_model(train_X[:d], train_X[d:d+30],
                                                              nw_weighted[:d], nw_weighted[d:d+30])

            # Predict and evaluate
            pred_val_w = pp_model_w.predict(test_X[:test_d])
            rmse_score_w, r2_score_w = model_eval(pred_val_w, nw_weighted_test[:test_d])

            rmse_w.append(rmse_score_w)
            r2_w.append(r2_score_w)

        preds2 = loaded_model2.predict(test_X[:test_d])
        plot_training(pp_hist_w)
        plot_predictions(pred_val_w, preds2, nw_weighted_test[:test_d], test_y[:test_d])

        figure, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(weights, rmse_w, label="RMSE weekend")
        ax[0].set_xlabel("Weights")
        ax[0].set_ylabel("RMSE")
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(weights, r2_w, label="R2-score weekend")
        ax[1].set_xlabel("Weights")
        ax[1].set_ylabel("R2 score")
        ax[1].grid()
        ax[1].legend()

        plt.show()

    if other:
        '''
        morning_shifts, evening_shifts, night_shifts = get_time_shifts(train_set)
        morning_shifts_val, evening_shifts_val, night_shifts_val = get_time_shifts(val_set)
        morning_shifts_test, evening_shifts_test, night_shifts_test = get_time_shifts(test_set)
    
        # Score data based on defined preferences regarding night shifts
        n_distr = night_distribution(night_shifts)
        n_scores = night_person(n_distr, night_shifts[0:q_answers])
        d_scores = non_night_person(n_distr, night_shifts[0:q_answers])
        n_scores, n_scores_val = train_test_split(n_scores, test_size=0.2, shuffle=False)
        d_scores, d_scores_val = train_test_split(d_scores, test_size=0.2, shuffle=False)
        # n_scores_val = night_person(n_distr, night_shifts_val[0:q_val])
        n_scores_test = night_person(n_distr, night_shifts_test)
        d_scores_test = non_night_person(n_distr, night_shifts_test)
        '''

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
