import pandas as pd
import numpy as np
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


# Count the number of consecutive working days in a roster for one person
def count_consecutive_working_days(data):
    max_days = []
    min_days = []
    for i in range(len(data)):  # Go through each row
        counter = 0
        max_v = 0
        min_v = 10
        for j in range(0, len(data[i]), 3):   # Go through each day
            w_day = 0
            for k in range(3):
                if data[i, j+k] == 1:   # Count if there is a shift
                    w_day = 1
                    break
            if w_day == 1:
                counter = counter + 1
                if counter > max_v:
                    max_v = counter
            elif (w_day == 0) and (counter > 0):
                if counter < min_v:
                    min_v = counter
                counter = 0
            else:
                counter = 0
        max_days.append(max_v)
        min_days.append(min_v)
    print(max_days, min_days)
    return max_days, min_days


# Score individual rosters based on the number of consecutive working days
def score_cons_days(max_days, min_days):
    score = []
    for (maxd, mind) in zip(max_days, min_days):
        if (mind > 2) and (maxd < 6):
            score.append(3)
        elif (mind == 2) or (maxd == 6):
            score.append(2)
        elif maxd == 7:
            score.append(1)
        elif (maxd > 7) or (mind == 1):
            score.append(0)
    return score


# Count the number of night shifts in a roster for one person
def count_night_shifts(data):
    nights = []
    for i in range(len(data)):
        n_shifts = 0
        for j in range(2, len(data[i]), 3):
            n_shifts = n_shifts + data[i, j]
        nights.append(n_shifts)
    return nights


# Score a roster for one person based on how many night shifts there are
def score_nr_nights(nights):
    score = []
    for n in nights:
        if n < 3:
            score.append(3)
        elif (n > 2) and (n < 5):
            score.append(2)
        elif (n > 4) and (n < 9):
            score.append(1)
        else:
            score.append(0)
    return score


if __name__ == '__main__':
    d_size = 5

    # Read data
    shift_data = read_data_df(d_size).values
    print(shift_data)
    # for p in range(len(shift_data)):
    # print(shift_data[p])

    # Score number of consecutive days
    max_nr_cons_days, min_nr_cons_days = count_consecutive_working_days(shift_data)
    cons_days_score = score_cons_days(max_nr_cons_days, min_nr_cons_days)

    # Score number of night shifts
    nr_night_shifts = count_night_shifts(shift_data)
    night_shift_score = score_nr_nights(nr_night_shifts)

    # Plot results
    figure, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].plot(max_nr_cons_days[0], cons_days_score[0], 'o')
    ax[0].plot(max_nr_cons_days[1], cons_days_score[1], 'x')
    ax[0].plot(max_nr_cons_days[2], cons_days_score[2], '<')
    ax[0].plot(max_nr_cons_days[3], cons_days_score[3], '>')
    ax[0].plot(max_nr_cons_days[4], cons_days_score[4], '^')
    ax[0].set_xlabel("Max number of consecutive days")
    ax[0].set_ylabel("Score")

    ax[1].plot(min_nr_cons_days[0], cons_days_score[0], 'o')
    ax[1].plot(min_nr_cons_days[1], cons_days_score[1], 'x')
    ax[1].plot(min_nr_cons_days[2], cons_days_score[2], '<')
    ax[1].plot(min_nr_cons_days[3], cons_days_score[3], '>')
    ax[1].plot(min_nr_cons_days[4], cons_days_score[4], '^')
    ax[1].set_xlabel("Min number of consecutive days")
    ax[1].set_ylabel("Score")

    ax[2].plot(nr_night_shifts[0], night_shift_score[0], 'o')
    ax[2].plot(nr_night_shifts[1], night_shift_score[1], 'x')
    ax[2].plot(nr_night_shifts[2], night_shift_score[2], '<')
    ax[2].plot(nr_night_shifts[3], night_shift_score[3], '>')
    ax[2].plot(nr_night_shifts[4], night_shift_score[4], '^')
    ax[2].set_xlabel("Number of night shifts")
    ax[2].set_ylabel("Score")
    plt.show()
