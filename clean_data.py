import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Create dataframe from given data
# Complexity O(N)
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
# Complexity O(1)
def split_to_values(row):
    arr_row = row.array
    rep1 = arr_row[0].replace("{", "")
    rep2 = rep1.replace("}", "")
    result = rep2.split(",")

    return result


# Read scores given for each shift
# Complexity O(N)
def read_scores(size):
    a = []
    f = open("scores.txt", "r")
    for i in range(size):
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
# TODO: improve complexity by introducing hash table
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


if __name__ == '__main__':
    # Complexity: N = num of rows being read
    # Complexity: M = num of elements in a row

    d_size = 3307321  # Data used for testing this script
    d_s = 3307321   # All data used for scoring
    num_shifts = 7  # A roster with fewer shifts than this will be removed

    ###############################
    #      Read general data      #
    ###############################

    # TODO: change data size to all data and check how many shifts are min shifts
    # Read data
    dataframe = read_data_df(d_size)
    gen_scores = np.array(read_scores(d_s))

    # Scale the scores
    scaler = MinMaxScaler()
    sc_scores = scaler.fit_transform(gen_scores.reshape(-1, 1))
    scores_f = sc_scores[:d_size]

    # Remove all rosters with fewer than 7 shifts
    dataframe, scores_f = remove_few_shifts(dataframe, scores_f, num_shifts)
    dataframe['score'] = scores_f

    '''
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
    # dataframe.to_csv('cleaned_df', index=False)

    # Check that the CSV loads
    new_df = pd.read_csv('cleaned_df')
    print(new_df.head(), len(new_df))
