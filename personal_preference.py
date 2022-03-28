import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from matplotlib.gridspec import GridSpec


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


# Remove rosters with too few shifts
def remove_few_shifts(data, scores):
    a = []
    s = []
    for row, sc in zip(data.values, scores):
        counter = 0
        for element in row:
            if element == 1.0:
                counter += 1
        if counter > 5:
            a.append(row)
            s.append(sc)
    df_new = pd.DataFrame(a)
    df_res = df_new.astype(float)
    return df_res, np.array(s)


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


#########################################
#      Train and evaluate networks      #
#########################################

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
    # for layer in loaded_model.layers[0:3]:
    # model.add(layer)

    model.add(loaded_model)
    # model.add(Dense(512, activation='relu', name="new_layer"))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='sigmoid', name="fine_tune"))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0)

    print(model.summary())

    return model, history


# Compile and fit CNN pp model (load pre-trained model and re-train it)
def pp_cnn_model(x_train, y_train):
    loaded_model = tf.keras.models.load_model("general_cnn_model_new")

    # Freeze some layers
    for layer in loaded_model.layers[:3]:
        layer.trainable = False

    # Check the trainable status of layers
    for layer in loaded_model.layers:
        print(layer, layer.trainable)

    model = Sequential(name="PPModelCNN")
    # for layer in loaded_model.layers[0:3]:
    # model.add(layer)

    model.add(loaded_model)
    # model.add(Dense(512, activation='relu', name="new_layer"))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='sigmoid', name="fine_tune"))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0)

    print(model.summary())

    return model, history


# Compile and fit pp model that is trained from scratch
def scratch_model(x_train, y_train):
    model = Sequential(name="PersonalPreferenceModel2")

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=20, verbose=0)

    print(model.summary())

    return model, history


# Compile and fit a convolutional neural network pp model from scratch
def cnn_pers_model(x_train, y_train):
    model = Sequential(name="CNN_model_PP")

    model.add(Conv2D(16, (2, 2), padding='same', activation="relu", input_shape=(9, 7, 1)))
    # model.add(MaxPooling2D((4, 4)))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.01, momentum=0.8), metrics=MeanAbsoluteError())
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=64, epochs=400, validation_split=0.1, verbose=0)

    return model, history


# Compile and fit a general convolutional neural network model
def cnn_model(x_train, y_train):
    model = Sequential(name="CNN_model_PP")

    history = LossHistory()

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
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.1, verbose=1, callbacks=[history])

    # model.save("general_cnn_model_new")

    return model, history


# check batch history of training general model on 1 epoch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))


# Evaluate model using RMSE and R^2
def model_eval(predicted, actual):
    mse = mean_squared_error(predicted, actual)
    rmse = np.sqrt(mse)
    print(rmse, type(actual), type(predicted))
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


# Get shifts in a format where weekday and week is visible
def get_graphic_schedule(data):
    columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df = []
    for line in data:
        shift = []
        for week in range(0, len(line), 21):
            shift_week = []
            for i in range(week, week+21, 3):
                val = "-"
                for j in range(3):
                    if j == 0:
                        if line[i+j] == 1:
                            val = "M"
                    if j == 1:
                        if (line[i+j] == 1) and (line[i+j-1] == 0):
                            val = "E"
                        elif (line[i+j] == 1) and (line[i+j-1] == 1):
                            val = "ME"
                    if j == 2:
                        if line[i + j] == 1:
                            val = "N"
                shift_week.append(val)
            shift.append(shift_week)
        dframe = pd.DataFrame(data=shift, columns=columns)
        df.append(dframe)
    return df


# Plot the graphic schedule
def plot_graphic_schedule(dfs, personal_score, loaded_preds, scratch_preds, pp_preds, cnn_preds):

    fig = plt.figure(figsize=(8, 6))

    gs = GridSpec(6, 2)  # 4 rows, 2 columns

    ax1 = fig.add_subplot(gs[:2, :])  # 2 first rows, both columns
    ax2 = fig.add_subplot(gs[2, 0])  # Third row, first column
    ax3 = fig.add_subplot(gs[3, 0])  # Fourth row, first column
    ax4 = fig.add_subplot(gs[4, 0])  # Fifth row, first column
    ax5 = fig.add_subplot(gs[2, 1])  # Third row, second column
    ax6 = fig.add_subplot(gs[3, 1])  # Fourth row, second column
    ax7 = fig.add_subplot(gs[4, 1])  # Fifth row, second column

    axes = [ax2, ax3, ax4, ax5, ax6, ax7]

    # create table
    for (df, ax, i) in zip(dfs, axes, range(len(dfs))):
        ax.axis('off')
        ax.axis('tight')
        ax.set_title('Shifts for person %i' %i)
        table = ax.table(cellText=df.values, colLabels=df.columns,
                         loc='center', cellLoc='center')

    ax1.plot(personal_score, 'o', color='blue', ls='-', label='Actual score')
    ax1.plot(loaded_preds, 's', color='red', ls='--', label='Only pre-trained model')
    ax1.plot(scratch_preds, 'x', color='green', ls='--', label='Only trained from scratch')
    ax1.plot(pp_preds, '^', color='purple', ls='--', label='Pre-trained model re-trained')
    ax1.plot(cnn_preds, 'v', color='orange', ls='--', label='Pre-trained CNN model re-trained')

    # display table
    fig.tight_layout()
    ax1.legend()
    plt.show()


###############################
#      General functions      #
###############################

# Set restored values to binary data (1 for shift 0 if not)
def set_binary(val):
    for i in range(len(val)):
        for j in range(len(val[i])-1):
            if val[i, j] < 0.5:
                val[i, j] = 0
            else:
                val[i, j] = 1
    return val


# Get number of different people who have answered
def get_num_people(data):
    data.rename(columns={1: 'Person', 0: 'Score'}, inplace=True)
    data['Person'] = data['Person'].astype('category')
    data['Person_category'] = data['Person'].cat.codes

    num_people, cols = data.groupby('Person').count().shape

    return data, num_people


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


# Plot the training history over epochs
def plot_training_batch(hists):
    figure, ax = plt.subplots(1, 1, figsize=(7, 6))

    for i in range(len(hists)-1):
        ax.plot(hists[i].history['loss'], color="blue")
        ax.plot(hists[i].history['val_loss'], color="orange")
        ax.set_xlabel("batch")
        ax.set_ylabel("loss")
        ax.grid()
        ax.legend()

    ax.plot(hists[len(hists)].history['loss'], color="blue", label="training")
    ax.plot(hists[len(hists)].history['val_loss'], color="orange", label="validation")

    plt.show()

    '''
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


if __name__ == '__main__':

    d_size = 200000  # How many shifts are used
    d_s = 3307321    # Read all scores (To scale them right)
    q_answers = 2000  # How many scored rosters there are
    test_d = 1000    # How many data points used for testing
    train_d = 50000

    comp_model = False  # Run comparisons between models
    general_cnn = False     # Train CNN on general scores
    cnn_personal = True    # Train CNN from scratch on only personal preference

    ###############################
    #      Read general data      #
    ###############################

    # Read data
    dataframe = read_data_df(d_size)
    gen_scores = np.array(read_scores(d_s))

    # Scale the scores
    scaler = MinMaxScaler()
    sc_scores = scaler.fit_transform(gen_scores.reshape(-1, 1))
    scores_f = sc_scores[:d_size]

    dataframe, scores_f = remove_few_shifts(dataframe, scores_f)

    # split data for training and testing
    train_X, test_X, train_y, test_y = train_test_split(dataframe.values, scores_f, test_size=0.2, shuffle=True)

    ###############################
    #      Read personal data     #
    ###############################

    # Read personal data
    p_data = read_personal_data('answers.txt')
    p_dfs = get_personal_data(p_data)
    pers_data = p_dfs[0]    # Choose which personal data to use

    pers_data = pers_data
    print(pers_data)

    # Get shifts and their score
    p_shifts = pers_data.iloc[:, 2:65].apply(pd.to_numeric)
    p_score = np.array(pers_data['Score'].apply(pd.to_numeric))

    # Scale score to a scale 0-1
    p_scaler = MinMaxScaler()
    p_scores = p_scaler.fit_transform(p_score.reshape(-1, 1))

    # Split data into train and test
    # pX_train, pX_test, py_train, py_test = train_test_split(p_shifts.values, p_scores, test_size=0.2)

    # Train CNN model on pp from scratch
    if cnn_personal:
        ###############################
        #      Training CNN           #
        ###############################
        hist_list = []

        for _ in range(20):

            # Split data into train and test
            pX_train, pX_test, py_train, py_test = train_test_split(p_shifts.values, p_scores, test_size=0.2)

            pers_train_X = get_shift_image(pX_train)
            pers_test_X = get_shift_image(pX_test)

            pers_train_X = np.array(pers_train_X)
            pers_test_X = np.array(pers_test_X)

            pers_train_X = pers_train_X.reshape(-1, 9, 7, 1)
            pers_test_X = pers_test_X.reshape(-1, 9, 7, 1)

            cnn_pers_mod, cnn_pers_hist = cnn_pers_model(pers_train_X[:train_d], py_train[:train_d])
            cnn_pers_pred = cnn_pers_mod.predict(pers_test_X)

            rmse_score, r2_score_ = model_eval(cnn_pers_pred, py_test)
            print("RMSE: ", rmse_score)
            print("R2: ", r2_score_)

            hist_list.append(cnn_pers_hist)

        plot_training_batch(hist_list)
        # plot_predictions(cnn_pers_pred, py_test)

    # Train CNN on general data
    if general_cnn:
        ###############################
        #      Training CNN           #
        ###############################

        gtrain_X = get_shift_image(train_X)
        gtest_X = get_shift_image(test_X)

        gtrain_X = np.array(gtrain_X)
        gtest_X = np.array(gtest_X)

        gtrain_X = gtrain_X.reshape(-1, 9, 7, 1)

        print(gtrain_X.shape)
        # print(gtest_X.shape)

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        cnn_mod, cnn_hist = cnn_model(gtrain_X[:train_d], train_y[:train_d])
        cnn_pred = cnn_mod.predict(gtest_X[:test_d])

        rmse_score, r2_score = model_eval(cnn_pred, test_y[:test_d])
        print("RMSE: ", rmse_score)
        print("R2: ", r2_score)

        plot_training_batch(cnn_hist)
        plot_predictions(cnn_pred, test_y[:test_d])

    # Run tests for comparing different types of models
    if comp_model:
        preds = 6

        # Load the pre-trained model
        loaded_model2 = tf.keras.models.load_model("g_scoring_model_new")
        l_pred = loaded_model2.predict(pX_test)
        rmse_score_l, r2_score_l = model_eval(l_pred, py_test)

        # Train model from scratch
        s_model, s_hist = scratch_model(pX_train, py_train)
        s_pred = s_model.predict(pX_test)
        rmse_score_s, r2_score_s = model_eval(s_pred, py_test)

        # Train MLP general model with personal preference
        pp_model, pp_hist = personal_preference_model(pX_train, py_train)
        pp_pred = pp_model.predict(pX_test)
        rmse_score_pp, r2_score_pp = model_eval(pp_pred, py_test)

        # Train CNN general model with personal preference
        gptrain_X = get_shift_image(pX_train)
        gptest_X = get_shift_image(pX_test)

        gptrain_X = np.array(gptrain_X)
        gptest_X = np.array(gptest_X)
        gptrain_X = gptrain_X.reshape(-1, 9, 7, 1)
        gptest_X = gptest_X.reshape(-1, 9, 7, 1)

        pp_model_cnn, pp_hist_cnn = pp_cnn_model(gptrain_X, py_train)
        pp_pred_cnn = pp_model_cnn.predict(gptest_X)
        rmse_score_cnn, r2_score_cnn = model_eval(pp_pred_cnn, py_test)

        # Print results
        print(test_X[:preds])
        print("Loaded model: RMSE: ", rmse_score_l, "R2: ", r2_score_l)
        print("Trained from scratch model: RMSE: ", rmse_score_s, "R2 : ", r2_score_s)
        print("Re-trained model: RMSE: ", rmse_score_pp, "R2: ", r2_score_pp)
        print("Re-trained model (CNN): RMSE: ", rmse_score_cnn, "R2: ", r2_score_cnn)

        dataframes = get_graphic_schedule(pX_test)
        plot_graphic_schedule(dataframes, py_test[:preds], l_pred[:preds],
                              s_pred[:preds], pp_pred[:preds], pp_pred_cnn[:preds])
