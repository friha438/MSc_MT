import pandas as pd
import numpy as np
from scipy.stats import rv_histogram, norm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


# Count how many night shifts each person has
def count_nights(data):
    nights = []
    for i in range(len(data)):
        counter = 0
        for j in range(2, len(data[i]), 3):
            if data[i, j] == 1:
                counter = counter + 1
        nights.append(counter)

    return nights


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
    score = np.random.normal(loc=static_score, scale=0.1)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score, n_nights


# Create a person to score good on few nights and bad on many
def non_night_person(distr, data):
    n_nights = count_nights(data)
    static_score_night = distr.cdf(n_nights)
    static_score = abs(np.array(static_score_night) - 1)
    score = np.random.normal(loc=static_score, scale=0.1)
    for i in range(len(score)):
        if score[i] > 1:
            score[i] = 1
        if score[i] < 0:
            score[i] = 0
    return score, n_nights


if __name__ == '__main__':
    d_size = 10000  # How many shifts are used

    # Read data
    dataframe = read_data_df(d_size)

    # split data for training, validation, and testing
    train_set, test_set = train_test_split(dataframe.values, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.1)

    n_distr = night_distribution(train_set)
    n_scores, n_night_shifts = night_person(n_distr, train_set[0:100])
    d_scores, d_night_shifts = non_night_person(n_distr, train_set[0:100])

    plt.plot(n_night_shifts, n_scores, 'o')
    plt.plot(n_night_shifts, d_scores, 'x')
    plt.show()

    '''
    fig, ax = plt.subplots()
    ax.grid(axis='y')
    plt.hist(night_shifts, bins=np.arange(12)-0.5, rwidth=0.8)
    plt.xticks(range(12))
    plt.xlabel("Number of night shifts in a roster per person")
    plt.ylabel("Distribution of night shifts in rosters")
    plt.show()
    '''

