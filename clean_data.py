import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Create dataframe from given data
# Complexity O(N)
def read_data_df(file):
    data = pd.read_fwf(file)
    d_f = data.iloc[1:, :]
    a = []
    for row in range(len(d_f)-1):
        r = split_to_values(d_f.iloc[row])
        a.append(r)

    df_new = pd.DataFrame(a)
    df_res = df_new.astype(float)
    return df_res


# Remove extra symbols and split each row to single int values
# Complexity O(1)
def split_to_values(row):
    arr_row = row.array
    rep1 = arr_row[0].replace("{", "")
    rep2 = rep1.replace("}", "")
    result = rep2.split(",")

    return result


# Read scores given for each shift
# Complexity O(N)
def read_scores(length, file):
    a = []
    f = open(file, "r")
    for i in range(length):
        r = float(f.readline())
        a.append(r)
    return a


# Remove rosters with too few shifts
# Complexity O(NxM)
def remove_few_shifts(data, scores, n):
    a = []
    s = []
    for row, sc in zip(data.values, scores):    # Iterate all data and their score
        counter = 0
        for element in row:     # Iterate all elements in a row
            if element == 1.0:
                counter += 1    # Count each shift in that roster
        if counter > n:         # If there are fewer than n shifts, remove them from the data
            a.append(row)
            s.append(sc)
    df_new = pd.DataFrame(a)
    df_res = df_new.astype(float)
    return df_res, np.array(s)


# Check for rows that are exactly the same, remove one of those
# Complexity O(N^2)
# TODO: improve complexity by introducing hash table?
def check_same(dataf):
    counter = 0
    same_shifts = []
    data = dataf.values
    for i in range(len(data)):      # Outer loop to iterate all rows in data
        for j in range(len(data)):  # Inner loop to iterate all rows in data
            if j != i:              # Avoid comparing to itself
                if (data[i, :62] == data[j, :62]).all():    # Check if all values in one row are the same as in another
                    same = [i, j]
                    same2 = [j, i]
                    # Check if this exact pair has already been found
                    if (same not in same_shifts) and (same2 not in same_shifts):
                        counter += 1   # Count each duplicate pair
                        print("SAME")
                        same_shifts.append(same)
    return counter, same_shifts


# Get distribution of all data
def get_data_distr(data):
    shifts_lst = []
    shifts_prop = []
    len_row = data.iloc[0, :]
    len_data = len(data)
    for i in range(len(len_row)-1):
        n_shifts = sum(data.iloc[:len_data, i])
        shifts_lst.append(n_shifts)
        shifts_prop.append(n_shifts/len_data)

    return shifts_lst, shifts_prop


if __name__ == '__main__':
    # Complexity: N = num of rows being read
    # Complexity: M = num of elements in a row

    ###############################
    #      Read general data      #
    ###############################

    # Read data
    data_file = 'shifts_new.txt'
    score_file = 'scores_new.txt'
    dataframe = read_data_df(data_file)
    gen_scores = np.array(read_scores(len(dataframe), score_file))

    # Scale the scores
    scaler = MinMaxScaler()
    sc_scores = scaler.fit_transform(gen_scores.reshape(-1, 1))

    # Remove all rosters with fewer than n shifts (only for old data)
    # num_shifts = 12  # A roster with fewer shifts than this will be removed
    # dataframe, scores_f = remove_few_shifts(dataframe, sc_scores, num_shifts)

    # Add scores and save data frame
    dataframe['score'] = sc_scores
    dataframe.to_csv('cleaned_df_new', index=False)

    # Check that the CSV loads
    new_df = pd.read_csv('cleaned_df_new')
    print(new_df.head(), len(new_df))

    # Plot distribution of data
    shifts, prop = get_data_distr(new_df)
    plt.plot(shifts)
    plt.show()

    '''
    # Computational complexity is too large to do this for the whole dataset, but can be done for <10000 data points
    # Check for rosters which are exactly the same and store index for those shifts
    c, s_shifts = check_same(dataframe)

    print(c)            # c = how many duplicates there are (might be removed later)
    print(s_shifts)     # s_shifts = an array of pairs where the indices are the same shifts

    # Print the first three pairs of same rosters + shape and size of dataframe
    print(dataframe.iloc[s_shifts[0], :].values)
    print(dataframe.iloc[s_shifts[1], :].values)
    print(dataframe.iloc[s_shifts[2], :].values)
    print(dataframe.head(), len(dataframe), len(dataframe)/d_size)

    # Get the first value in each pair of twins (the index which should be removed)
    shifts = []
    for sh in range(len(s_shifts)):
        print(s_shifts[sh][0])
        if s_shifts[sh][0] not in shifts:     # Take care of shift indices where there are more than 2 twin shifts
            shifts.append(s_shifts[sh][0])

    # Drop the rows with the same values as some other row and save into csv
    dataframe.drop(shifts, inplace=True)
    dataframe.to_csv('new_df', index=False)
    print(dataframe.head(), len(dataframe), len(dataframe)/d_size)
    '''
