import pandas as pd
import numpy as np
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import SGD
from matplotlib import style


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
        for j in range(0, len(data[i]), 3):  # Go through each day (3 shifts)
            for k in range(3):
                if k == 0:
                    m_shifts.append(data[i, j])
                elif k == 1:
                    e_shifts.append(data[i, j + k])
                else:
                    n_shifts.append(data[i, j + k])
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
    shifts_week = 21  # How many shifts there are in 1 week
    weekday_shifts = []
    for i in range(len(data)):
        counter = 0
        for j in range(start, end, 1):
            if data[i, j] == 1:
                counter = counter + 1
        for j in range(start + shifts_week, end + shifts_week, 1):
            if data[i, j] == 1:
                counter = counter + 1
        for j in range(start + 2 * shifts_week, end + 2 * shifts_week, 1):
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
        return 0, 5  # As a default check Monday and Tuesday


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
def weigh_scores(g_scores, p_scores, weight=0.5):
    f_scores = []
    for i in range(len(g_scores)):
        f_scores.append(weight * p_scores[i] + (1 - weight) * g_scores[i])
    return np.array(f_scores)


#########################################
#      Train and evaluate networks      #
#########################################

# Compile and fit pp model
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
    # model.add(Dense(512, activation='relu', name="new_layer"))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='sigmoid', name="fine_tune"))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, validation_split=0.1)

    print(model.summary())

    return model, history


# Compile and fit pp model that is trained from scratch
def scratch_model(x_train, y_train):
    model = Sequential(name="PersonalPreferenceModel2")

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0)

    print(model.summary())

    return model, history


# Evaluate model using RMSE and R^2
def model_eval(predicted, actual):
    mse = mean_squared_error(predicted, actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return rmse, r2


################################
#      Plots and graphics      #
################################

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
def plot_predictions(pred1, test1):
    vals = 16
    figure, ax = plt.subplots(1, 1, figsize=(5, 5))
    major_ticks = np.arange(0, 16, 1)

    ax.plot(pred1[0:vals], 'o', label="predicted")
    ax.plot(test1[0:vals], 'x', label="real")
    ax.set_title("Predicted vs real night weighted scores")
    ax.set_xticks(major_ticks)
    ax.grid(which="major", linestyle='--')
    ax.set_xlim(-1, vals)
    ax.set_ylim(-0.01, 1.01)
    ax.legend()
    '''
    ax[1].plot(pred2[0:vals], 'o', label="predicted")
    ax[1].plot(test2[0:vals], 'x', label="real")
    ax[1].set_title("Predicted vs real unweighted data")
    ax[1].set_xticks(major_ticks)
    ax[1].grid(which="major", linestyle='--')
    ax[1].set_xlim(-1, vals)
    ax[1].set_ylim(-0.01, 1.01)
    ax[1].legend()
    '''
    plt.show()


###############################
#      General functions      #
###############################

# Set restored values to binary data (1 for shift 0 if not)
def set_binary(val):
    for i in range(len(val)):
        for j in range(len(val[i]) - 1):
            if val[i, j] < 0.5:
                val[i, j] = 0
            else:
                val[i, j] = 1
    return val


if __name__ == '__main__':

    d_size = 200000  # How many shifts are used
    q_answers = 2000  # How many scored rosters there are
    weight_wishes = 0.5    # Weigh scores between two preferences
    PP_model = True  # Train, predict and evaluate a pp model for a given amount of data and weight distribution
    PP_model_iter = False  # Same as PP_model, but for a set of weight distributions and different amount of data
    plot_distributions = False  # Plots the scores and distribution of shifts

    ###############################
    #      Read general data      #
    ###############################

    df = pd.read_csv('cleaned_df')
    X = df.iloc[:d_size, :63].values
    y = df.iloc[:d_size, 63].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True)

    ###############################
    #      Score shifts on PP     #
    ###############################

    # Get different shifts of the day
    morning_shifts, evening_shifts, night_shifts = get_time_shifts(train_X)
    morning_shifts_test, evening_shifts_test, night_shifts_test = get_time_shifts(test_X)

    # Score data based on liking/disliking nights
    n_distr, number_nights = night_distribution(night_shifts)
    n_distr_test, number_nights_test = night_distribution(night_shifts_test)
    n_scores = negative_corr_person(n_distr, number_nights[:q_answers])
    n_scores_test = negative_corr_person(n_distr, number_nights_test[:q_answers])

    # Score data based on liking/disliking weekends
    w_dist, w_shifts = weekend_distribution(train_X)
    w_dist_test, w_shifts_test = weekend_distribution(test_X)
    w_scores = negative_corr_person(w_dist, w_shifts[:q_answers])
    w_scores_test = negative_corr_person(w_dist, w_shifts_test[:q_answers])

    # Score data based on liking/disliking 2 specific consecutive days
    wd_dist, wd_shifts = weekday_distribution(train_X, "wed_thu")
    wd_dist_test, wd_shifts_test = weekday_distribution(test_X, "wed_thu")
    wd_scores = negative_corr_person(wd_dist, wd_shifts[:q_answers])
    wd_scores_test = negative_corr_person(wd_dist, wd_shifts_test[:q_answers])

    # Weigh preferences to have more complex wishes
    weighted_sc = weigh_scores(n_scores, w_scores, weight=weight_wishes)
    weighted_sc_test = weigh_scores(n_scores_test, w_scores_test, weight=weight_wishes)

    if PP_model:
        d = 200  # Data points used for training
        w = 0.9  # Weight between general scores and preference scores

        final_weighted = weigh_scores(train_y[:d], w_scores[:d], weight=w)
        final_weighted_test = weigh_scores(test_y[:q_answers], w_scores_test[:q_answers], weight=w)

        # Create and train model
        pp_model, pp_hist = personal_preference_model(train_X[:d], final_weighted)

        # Predict and evaluate
        pred_val = pp_model.predict(test_X[:q_answers])
        rmse_score, r2_score = model_eval(pred_val, final_weighted_test)

        print("RMSE (weekend person): ", rmse_score)
        print("R2 (weekend person): ", r2_score)

        plot_training(pp_hist)

    if PP_model_iter:
        weight_step = 0.2
        data_step = 20
        weights = np.arange(0.0, 1.05, weight_step)
        data_used = np.arange(10, 111, data_step)

        rmse_lst = []
        r2_lst = []
        x_weights = []
        y_data = []

        for w in weights:
            if len(weights) != len(data_used):
                print("weights and data used are of different lengths!")
                break
            for d in data_used:
                print("Weight: ", w, "Data used:", d)
                final_weighted = weigh_scores(train_y[:q_answers], n_scores, weight=w)
                final_weighted_test = weigh_scores(test_y[:q_answers], n_scores_test, weight=w)

                # Create and train model
                pp_model, pp_hist = personal_preference_model(train_X[:d], final_weighted[:d])

                # Predict and evaluate
                pred_val = pp_model.predict(test_X[:q_answers])
                rmse_score, r2_score = model_eval(pred_val, final_weighted_test[:q_answers])

                x_weights.append(w)
                y_data.append(d)
                rmse_lst.append(rmse_score)
                r2_lst.append(r2_score)

        ############################################
        #       Plot 3D graph of evaluations       #
        ############################################

        eval_res = np.array(r2_lst)     # evaluation being plotted
        # eval_res = np.array(rmse)     # evaluation being plotted

        style.use('ggplot')
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')

        x3 = np.array(x_weights) - weight_step / 2
        y3 = np.array(y_data) - data_step / 2
        z3 = np.zeros(len(x_weights))

        dx = weight_step * np.ones(len(x_weights))
        dy = data_step * np.ones(len(x_weights))
        dz = eval_res

        ax1.bar3d(x3, y3, z3, dx, dy, dz)

        ax1.set_xlabel('Weights')
        ax1.set_ylabel('Data used')
        ax1.set_zlabel('R2')            # Change according to evaluation method

        plt.show()

    if plot_distributions:
        # Plot distribution of scores based on given weights
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(number_nights[:len(weighted_sc)], weighted_sc, 'o')
        axes[1].plot(w_shifts[:len(weighted_sc)], weighted_sc, 'x')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.grid(axis='y')
        plt.hist(number_nights, bins=np.arange(12)-0.5, rwidth=0.8)
        plt.xticks(range(12))
        plt.xlabel("Number of night shifts in a roster per person")
        plt.ylabel("Distribution of night shifts in rosters")
        plt.show()
