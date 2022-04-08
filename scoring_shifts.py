import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#############################
#       Read data           #
#############################

# Create dataframe from given data
def read_data_df():
    data = pd.read_fwf('shifts_new.txt')
    d_f = data.iloc[1:, :]
    a = []
    for row in range(len(d_f)-1):
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


####################################
#       General functions          #
####################################

# Get a list that only displays night shifts
def get_night_shifts(data):
    nights = []
    for i in range(len(data)):
        n_shifts = []
        for j in range(2, len(data[i]), 3):
            n_shifts.append(data[i, j])
        nights.append(n_shifts)
    return nights


# Plots scores from consecutive days and night shifts
def plot_2methods(max_nr_cons_days, min_nr_cons_days, cons_days_score, nr_night_shifts, night_shift_score):
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


###########################################
#       1. Consecutive working days       #
###########################################
# NOTE: No shifts have more than 6 consecutive days, but many have only 1 day of work
# However some have 2 shifts in that day, and some might have more than 1 of these kinds of shifts

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


###########################################
#       2. Number of night shifts         #
###########################################

# Count the number of night shifts in a roster for one person
def count_night_shifts(data):
    nights = get_night_shifts(data)
    n_nights = []
    for i in range(len(nights)):
        n_nights.append(np.sum(nights[i]))

    return n_nights


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


###########################################
#       3. Consecutive night shifts       #
###########################################

# Count the max number of consecutive nights in a roster for one person
def count_consecutive_nights(nights):
    max_nights = []
    for i in range(len(nights)):  # Go through each row
        counter = 0
        max_n = 0
        for j in range(len(nights[i])):   # Go through each day
            if nights[i, j] == 1:
                counter = counter + 1
                if counter > max_n:
                    max_n = counter
            else:
                counter = 0
        max_nights.append(max_n)
    return max_nights


# Score a roster for one person based on how many consecutive night shifts there are
def score_cons_nights(c_nights):
    score = []
    for n in c_nights:
        if n < 3:
            score.append(3)
        elif (n > 2) and (n < 5):
            score.append(2)
        elif n == 5:
            score.append(1)
        else:
            score.append(0)
    return score


###########################################
#     4. Consecutive shifts (hours)       #
###########################################

# Count the max number of consecutive working days (in hours) in a roster for one person
def count_consecutive_hours(data):
    cons_hours = []
    for i in range(len(data)):  # Go through each row
        counter = 0
        max_h = 0
        for j in range(0, len(data[i]), 3):   # Go through each day
            w_day = 0
            for k in range(3):
                if (k == 2) and (data[i, j+k] == 1):
                    counter = counter + 12
                    w_day = 1
                elif data[i, j+k] == 1:
                    counter = counter + 8
                    w_day = 1
            if w_day == 1:
                if counter > max_h:
                    max_h = counter
            else:
                counter = 0
        cons_hours.append(max_h)
    return cons_hours


# Score a roster for one person based on how many consecutive shifts there are (in hours)
def score_cons_hours(cons_hours):
    score = []
    for h in cons_hours:
        if h < 41:
            score.append(3)
        elif (h > 40) and (h < 49):
            score.append(2)
        elif (h > 48) and (h < 56):
            score.append(1)
        else:
            score.append(0)
    return score


###########################################
#        5. Night shift recovery          #
###########################################

# Count the minimum number of hours of rest after a night shift
def count_recovery(data):
    min_rec = []
    for i in range(len(data)):  # Go through each row
        counter = 0
        n_shift = 0
        min_r = 100
        morn_rec = 0
        night_rec = 0
        for j in range(0, len(data[i]), 3):  # Go through each day
            for k in range(3):  # Go through morning, evening, night shifts
                if (k == 0) and (n_shift == 1):     # If there has been a night shift and now is morning
                    if data[i, j+k] == 1:           # Check if there is a morning shift
                        n_shift = 0                 # Stop adding recovery
                        break
                    if (data[i, j+k] == 1) and (data[i, j+k-1] == 1):    # If morning after night
                        morn_rec = 1
                    else:
                        counter = counter + 4.5     # Add recovery
                elif (k == 1) and (n_shift == 1):   # If there has been a night shift and now is evening
                    if data[i, j+k] == 1:           # Check if there is an evening shift
                        n_shift = 0                 # Stop adding recovery
                        break
                    else:
                        counter = counter + 7.5     # Add recovery
                elif (k == 2) and (n_shift == 1):   # If there has been a night shift and now is night
                    if (data[i, j+k] == 1) and (data[i, j+k-3] == 1):  # Night shift tonight and yesterday
                        counter = 0
                    elif (data[i, j+k] == 1) and (data[i, j+k-3] == 0):  # Night shift tonight but not yesterday
                        night_rec = counter
                        break
                    elif data[i, j+k] == 0:         # And there is no night shift tonight
                        counter = counter + 11.75       # Add recovery
                if (k == 2) and (data[i, j+k] == 1):    # If it is a night shift, mark that
                    n_shift = 1

            if n_shift == 0:
                if (counter > 0) and (counter < min_r):
                    min_r = counter
                counter = 0
            elif night_rec > 0:         # Check special case of ending with night shift and starting with night shift
                if night_rec < min_r:
                    min_r = night_rec
                night_rec = 0
            elif morn_rec == 1:         # Check special case of working a morning shift after a night shift
                min_r = morn_rec

        min_rec.append(min_r)
    return min_rec


# Score a roster for one person based on how much recovery there is after night shifts
def score_recovery(min_rec):
    score = []
    for h in min_rec:
        if h > 48:
            score.append(3)
        elif (h > 27) and (h < 49):
            score.append(2)
        elif (h > 10) and (h < 28):
            score.append(1)
        else:
            score.append(0)
    return score


###############################
#        6. Weekends          #
###############################

# Count how many weekend shifts there are in each roster for one person
# NOTE: At the moment Friday night shifts are included in weekends
def count_weekends(data):
    weekends = []
    for i in range(len(data)):
        w1 = 0
        w2 = 0
        w3 = 0
        for j in range(14, 20, 1):
            if data[i, j] == 1:
                w1 = 1
                break
        for j in range(35, 41, 1):
            if data[i, j] == 1:
                w2 = 1
                break
        for j in range(56, 62, 1):
            if data[i, j] == 1:
                w3 = 1
                break
        weekends.append(w1 + w2 + w3)
    return weekends


# Score a roster for one person based on how much recovery there is after night shifts
def score_weekends(weekends):
    score = []
    for w in weekends:
        if w < 2:
            score.append(3)
        elif w == 2:
            score.append(2)
        else:
            score.append(1)
    return score


###############################
#        Final score          #
###############################

# Score sum of all scores
def score_shifts(cd_sc, ns_sc, cn_sc, ch_sc, r_sc, w_sc):
    score = []
    for i in range(len(cd_sc)):
        sc = cd_sc[i] + ns_sc[i] + cn_sc[i] + ch_sc[i] + r_sc[i] + w_sc[i]
        score.append(sc/6)
    return score


# Save scores to text file
def save_scores(scores):
    scores = scores.reshape((len(scores), 1))
    textfile = open("scores_new.txt", "w")
    for sc in scores:
        np.savetxt(textfile, sc)
    textfile.close()


if __name__ == '__main__':
    # d_size = 3307321
    plot_dist1 = False   # Plots the distribution of scores 1
    plot_dist2 = False   # Plots the distribution of scores 2
    final_dist = False   # Plots distribution of final score

    # Read data
    shift_data = read_data_df().values

    # Score number of consecutive days
    max_cons_days, min_cons_days = count_consecutive_working_days(shift_data)
    cons_days_scores = score_cons_days(max_cons_days, min_cons_days)

    # Score number of night shifts
    night_shifts = count_night_shifts(shift_data)
    night_shift_scores = score_nr_nights(night_shifts)

    # Score number of consecutive night shifts
    nights_n = np.array(get_night_shifts(shift_data))
    c_night_shifts = count_consecutive_nights(nights_n)
    cons_nights_scores = score_cons_nights(c_night_shifts)

    # Score number of consecutive shifts (in hours)
    cons_days_h = count_consecutive_hours(shift_data)
    cons_hours_scores = score_cons_hours(cons_days_h)

    # Score recovery (in hours) after night shifts
    recovery_hours = count_recovery(shift_data)
    recovery_scores = score_recovery(recovery_hours)

    # Score how many free weekends there are
    weekend_shifts = count_weekends(shift_data)
    weekend_scores = score_weekends(weekend_shifts)

    final_score = score_shifts(cons_days_scores, night_shift_scores, cons_nights_scores,
                               cons_hours_scores, recovery_scores, weekend_scores)

    save_scores(np.array(final_score))

    if plot_dist1:
        fig1, ax1 = plt.subplots(1, 3, figsize=(14, 5))
        ax1[0].grid(axis='y')
        ax1[0].hist(cons_days_scores, bins=np.arange(5) - 0.5, rwidth=0.8)
        ax1[0].set_xlabel("Scores for consecutive days")
        ax1[0].set_ylabel("Distribution of scores")

        ax1[1].grid(axis='y')
        ax1[1].hist(cons_hours_scores, bins=np.arange(5) - 0.5, rwidth=0.8)
        ax1[1].set_xlabel("Scores for consecutive shifts (in hours)")

        ax1[2].grid(axis='y')
        ax1[2].hist(weekend_scores, bins=np.arange(5) - 0.5, rwidth=0.8)
        ax1[2].set_xlabel("Scores for free weekends")
        plt.show()

    if plot_dist2:
        fig2, ax2 = plt.subplots(1, 3, figsize=(14, 5))
        ax2[0].grid(axis='y')
        ax2[0].hist(night_shift_scores, bins=np.arange(5) - 0.5, rwidth=0.8)
        ax2[0].set_xlabel("Scores for night shifts")
        ax2[0].set_ylabel("Distribution of scores")

        ax2[1].grid(axis='y')
        ax2[1].hist(cons_nights_scores, bins=np.arange(5) - 0.5, rwidth=0.8)
        ax2[1].set_xlabel("Scores for consecutive night shifts")

        ax2[2].grid(axis='y')
        ax2[2].hist(recovery_scores, bins=np.arange(5) - 0.5, rwidth=0.8)
        ax2[2].set_xlabel("Scores for recovery hours")
        plt.show()

    if final_dist:
        fig, axis = plt.subplots()
        axis.grid(axis='y')
        plt.hist(final_score, bins=np.arange(0, 4, 1 / 6) - 1 / (2 * 6), rwidth=0.8)
        plt.xticks(np.arange(0, 4, 0.5))
        plt.xlabel("Score of the roster")
        plt.ylabel("Distribution of scores")
        plt.show()

    # Check if methods work correctly
    # plot_2methods(max_cons_days, min_cons_days, cons_days_scores, night_shifts, night_shift_scores)
