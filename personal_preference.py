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

from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


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

    return night_dist, n_nights


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


##################################################
#      2 Consecutive days shift preferences      #
##################################################

# Count how many shifts of 2 chosen consecutive days a person has worked
def count_weekdays(data, start, end):
    shifts_week = 21    # How many shifts there are in 1 week
    weekday_shifts = []
    for i in range(len(data)):
        counter = 0
        for j in range(start, end, 1):
            if data[i, j] == 1:
                counter = counter + 1
        for j in range(start + shifts_week, end + shifts_week, 1):
            if data[i, j] == 1:
                counter = counter + 1
        for j in range(start + 2*shifts_week, end + 2*shifts_week, 1):
            if data[i, j] == 1:
                counter = counter + 1
        weekday_shifts.append(counter)
    return weekday_shifts


def get_day_index(days):
    if days == "tue_wed":
        return 3, 8
    elif days == "wed_thu":
        return 6, 11
    elif days == "thu_fri":
        return 9, 14
    else:
        return 0, 5     # As a default check Monday and Tuesday


# Find the distribution of 2 chosen consecutive days
def weekday_distribution(data, days):
    start, end = get_day_index(days)
    weekday_shifts = count_weekdays(data, start=start, end=end)
    weekday_hist = np.histogram(weekday_shifts, bins=np.arange(11))
    weekday_dist = rv_histogram(weekday_hist)
    return weekday_dist, weekday_shifts


#########################################
#      Create and weigh preferences     #
#########################################

# Create a person that gives good scores for many shifts of one type
def positive_corr_person(dist, n_shifts):
    static_score = dist.cdf(n_shifts)
    score = np.random.normal(loc=static_score, scale=0.005)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


# Create a person that gives poor scores for many shifts of one type
def negative_corr_person(dist, n_shifts):
    static_score = dist.cdf(n_shifts)
    score_inv = abs(np.array(static_score) - 1)
    score = np.random.normal(loc=score_inv, scale=0.005)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score


# Weigh the two types of scoring to create new scoring
def weigh_scores(g_scores, p_scores, w=0.5):
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
    history = model.fit(x_train, y_train, batch_size=64, epochs=500, validation_data=(x_val, y_val), verbose=0)

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
    weight = 0.3
    get_preferences = True
    PP_model = False    # If set to true, then get_preferences also needs to be true
    PP_model_iter = True    # If set to true, then get_preferences also needs to be true

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
        # Get different shifts of the day
        morning_shifts, evening_shifts, night_shifts = get_time_shifts(train_X)
        morning_shifts_test, evening_shifts_test, night_shifts_test = get_time_shifts(test_X)

        # Score data based on liking nights
        n_distr, number_nights = night_distribution(night_shifts)
        n_distr_test, number_nights_test = night_distribution(night_shifts_test)
        n_scores = positive_corr_person(n_distr, number_nights[:q_answers])
        # n_scores, n_scores_val = train_test_split(n_scores, test_size=0.2, shuffle=False)
        n_scores_test = positive_corr_person(n_distr, number_nights_test[:test_d])

        # Score data based on liking weekends
        w_dist, w_shifts = weekend_distribution(train_X)
        w_dist_test, w_shifts_test = weekend_distribution(test_X)
        w_scores = positive_corr_person(w_dist, w_shifts[:q_answers])
        # w_scores, w_scores_val = train_test_split(w_scores, test_size=0.2, shuffle=False)
        w_scores_test = positive_corr_person(w_dist, w_shifts_test[:test_d])

        # Score data based on liking 2 specific consecutive days
        wd_dist, wd_shifts = weekday_distribution(train_X, "tue_wed")
        wd_dist_test, wd_shifts_test = weekday_distribution(test_X, "tue_wed")
        wd_scores = positive_corr_person(wd_dist, wd_shifts[:q_answers])
        # wd_scores, wd_scores_val = train_test_split(wd_scores, test_size=0.2, shuffle=False)
        wd_scores_test = positive_corr_person(wd_dist, wd_shifts_test[:test_d])

        # Weigh preferences to have more complex wishes
        weighted = weigh_scores(wd_scores, n_scores, w=weight)
        weighted_test = weigh_scores(wd_scores_test, n_scores_test, w=weight)

        # Plot distribution of scores based on given weights
        # figure, ax = plt.subplots(1, 2, figsize=(12, 5))
        # ax[0].plot(number_nights[:len(weighted)], weighted, 'o')
        # ax[1].plot(wd_shifts[:len(weighted)], weighted, 'x')
        # plt.show()

    if PP_model:
        '''
        d = 100     # Data points used for training
        d_val = 30  # How many data points are used for validation

        final_weighted = weigh_scores(train_y[:d], weighted[:d], w=weight)
        final_weighted_val = weigh_scores(train_y[d:d+d_val], weighted[d:d+d_val], w=weight)
        final_weighted_test = weigh_scores(test_y[:test_d], weighted_test[:test_d], w=weight)

        # Create and train model
        pp_model_w, pp_hist_w = personal_preference_model(train_X[:d], train_X[d:d+d_val],
                                                          final_weighted, final_weighted_val)

        # Predict and evaluate
        pred_val_w = pp_model_w.predict(test_X[:test_d])
        rmse_score_w, r2_score_w = model_eval(pred_val_w, final_weighted_test)

        print("Execution time: ", datetime.now() - startTime)
        print("RMSE (weekend person): ", rmse_score_w)
        print("R2 (weekend person): ", r2_score_w)

        plt.plot(w_shifts[:len(final_weighted)], final_weighted, 'o')
        plt.show()

        preds2 = loaded_model2.predict(test_X[:test_d])
        plot_training(pp_hist_w)
        plot_predictions(pred_val_w, preds2, final_weighted_test[:test_d], test_y[:test_d])
        '''
    if PP_model_iter:
        weight_step = 0.1
        data_step = 50
        weights = np.arange(0.0, 1.05, weight_step)
        data_used = np.arange(50, 551, data_step)

        rmse_w = []
        r2_w = []
        x_weights = []
        y_data = []

        for w in weights:
            if len(weights) != len(data_used):
                print("weights and data used are of different lengths!")
                break
            for d in data_used:
                print("Weight: ", w, "Data used:", d)
                final_weighted = weigh_scores(train_y[:q_answers], n_scores, w=w)
                final_weighted_test = weigh_scores(test_y[:test_d], n_scores_test, w=w)

                # Create and train model
                pp_model_w, pp_hist_w = personal_preference_model(train_X[:d], train_X[d:d+30],
                                                                  final_weighted[:d], final_weighted[d:d+30])

                # Predict and evaluate
                pred_val_w = pp_model_w.predict(test_X[:test_d])
                rmse_score_w, r2_score_w = model_eval(pred_val_w, final_weighted_test[:test_d])

                x_weights.append(w)
                y_data.append(d)
                rmse_w.append(rmse_score_w)
                r2_w.append(r2_score_w)

        rmse_res = np.array(rmse_w)

        style.use('ggplot')
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        x3 = np.array(x_weights)-weight_step/2
        y3 = np.array(y_data)-data_step/2
        z3 = np.zeros(len(x_weights))

        dx = weight_step*np.ones(len(x_weights))
        dy = data_step*np.ones(len(x_weights))
        dz = rmse_res

        ax1.bar3d(x3, y3, z3, dx, dy, dz)

        ax1.set_xlabel('Weights')
        ax1.set_ylabel('Data used')
        ax1.set_zlabel('RMSE')

        plt.show()

    other = False
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

        '''
        preds2 = loaded_model2.predict(test_X[:test_d])
        plot_training(pp_hist_w)
        plot_predictions(pred_val_w, preds2, final_weighted_test[:test_d], test_y[:test_d])

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
        '''
