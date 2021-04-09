import copy

import numpy as np
import pandas as pd


def smooth_speed(speed, win_size=2):
    df = pd.DataFrame(speed)
    smooth_speed = (df.rolling(window=win_size, min_periods=1).mean()).values
    return smooth_speed


def neg_values(array):
    new_array = copy.copy(array)
    for index, element in enumerate(array):
        if not np.isnan(element):
            new_array[index] = -abs(element)
    return new_array


def remove_high_spd(speed_raw):
    # takes the raw speed and will replace any value over threshold with the mean of the values < threshold -5:+5.
    # This is to get rid of the massive speed jumps caused by roi jumps
    speed_t = copy.copy(speed_raw)
    threshold = np.nanpercentile(speed_raw, 95) * 2
    ind_high = np.where(speed_raw > threshold)[0]

    # for each index > threshold find the values on the side and take the average of all of those values < threshold
    for index_n in range(0, ind_high.shape[0]):
        win_min = ind_high[index_n] - 5
        win_max = ind_high[index_n] + 5

        if win_min < 0:
            win_min = 0
        if win_max > speed_t.shape[0]:
            win_max = speed_t.shape[0]

        values = speed_t[win_min:win_max]
        speed_t[ind_high[index_n]] = np.nanmean(values[values < threshold])
    return speed_t


def binner(input_data, bin_width, axis_d):
    """ takes a 1D np array and bins it with bin size of bin_width, input needs to be with data in dim 0 e.g.
    [10,0] or [10,] """
    # input must be np.array, if pandas.series change it to np.array
    if isinstance(input_data, pd.core.series.Series):
        print("correcting input from pd.series to np.array")
        input_data = input_data.to_numpy()

    if input_data.shape[0] % bin_width != 0:
        rest = input_data.shape[0] % bin_width
        output_data = np.reshape(input_data[0:-rest], [int(input_data[0:-rest].shape[0] / bin_width), bin_width])
    else:
        output_data = np.reshape(input_data, [int(input_data.shape[0] / bin_width), bin_width])

    output_data_mean = np.nanmean(output_data, axis=axis_d)

    return output_data_mean, output_data