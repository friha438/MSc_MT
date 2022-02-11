import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsoluteError
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


# Read scores given for each shift
def read_scores(size):
    a = []
    f = open("scores.txt", "r")
    for i in range(size):
        r = float(f.readline())
        a.append(r)
    return a


# Compile and fit general model
def general_model(x_tr, x_val, y_tr, y_val):
    model = Sequential(name="GeneralModel")
    model.add(Dense(512, activation='relu'))
    # for i in range(4):
    #    model.add(Dense(256, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=MeanAbsoluteError())
    history = model.fit(x_tr, y_tr, batch_size=64, epochs=15, validation_data=(x_val, y_val), verbose=1)

    print(model.summary())

    return model, history


# Plot the training history over epochs
def plot_training(hist):
    figure, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(hist.history['loss'], label="training")
    ax[0].plot(hist.history['val_loss'], label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Loss")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(hist.history['mean_absolute_error'], label="training")
    ax[1].plot(hist.history['val_mean_absolute_error'], label="validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mean absolute error")
    ax[1].set_title("Measure of accuracy")
    ax[1].grid()
    ax[1].legend()
    plt.show()


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


if __name__ == '__main__':
    startTime = datetime.now()  # For measuring execution time
    d_size = 1000000   # How much data to use

    # Read data
    shift_data = read_data_df(d_size).values
    shift_scores = np.array(read_scores(d_size))

    # Scale the scores
    scaler = MinMaxScaler()
    sc_scores = scaler.fit_transform(shift_scores.reshape(-1, 1))

    # split data for training, validation, and testing
    X_train, X_test, y_train, y_test = train_test_split(shift_data, sc_scores, test_size=0.2, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    # Create and train model
    g_model, g_history = general_model(X_train, X_valid, y_train, y_valid)
    y_pred = g_model.predict(X_test)

    # Evaluate model
    rmse_sc, r2_sc = model_eval(y_pred, y_test)
    print("RMSE: ", rmse_sc)
    print("R2: ", r2_sc)
    print("Execution time: ", datetime.now() - startTime)

    plot_training(g_history)
    plot_predictions(y_pred, y_test)
