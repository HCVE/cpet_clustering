import numpy as np
import warnings
import pandas as pd

warnings.simplefilter("ignore")


def check_for_missing_values(dataframe):
    """This function checks for extreme low (near zero) values
    at the beginning of the CPET recordings and removes them"""
    dataframe = dataframe.fillna(0)
    # find the index of the first non-zero element in the data
    zero_at_beginning = 0
    for counter, d in enumerate(dataframe["V'CO2"]):
        if float(d) <= 80:
            zero_at_beginning = counter
        else:
            break
    return dataframe.iloc[zero_at_beginning + 1:]


def check_for_ending_zeros(dataframe):
    """This function checks for extreme low (near zero) values
        at the end of the CPET recordings and removes them"""

    variables = ["PETCO2", "PETO2", "PECO2", "PEO2", "V'CO2", "V'O2", "V'E", "RER"]
    zeros_at_end = len(dataframe) - 1
    max_pos = np.argmax(dataframe["V'CO2"].astype(float))
    for counter, d in reversed(list(enumerate(dataframe["V'CO2"]))):
        if float(d) <= 350:
            zeros_at_end = counter
        else:
            break
    for var in variables:
        try:
            dataframe[var].iloc[zeros_at_end:] = dataframe[var].iloc[max_pos]
        except KeyError:
            continue
    return dataframe


def check_for_intermediate_errors(dataframe):
    """This function checks for extreme changes in the middle of the CPET recordings
    and corrects them by replacing them with the sum of the previous value and the absolute half difference"""

    variables_to_check = ["HR", "PETO2", "PEO2", "PETCO2", "PECO2", "V'CO2", "V'O2"]

    for label in variables_to_check:
        d = np.array(dataframe[label].astype(float))
        difference = np.abs(np.diff(d))
        difference = np.round((difference - np.mean(difference))/np.std(difference), 1)
        pos = np.where(difference >= 3)[0]

        # if len(pos) == 2 and pos[0] != 0:
        if len(pos) == 2 and pos[0] >= 2:
            val = correct_values(d, pos)
            dataframe[label] = val
    return dataframe


def correct_values(data, pos):
    if np.diff(pos) <= 10:
        for i in range(pos[0], pos[-1]+1):
            if i > 2:
                data[i] = data[i-1] + 0.5 * (abs(data[i-1] - data[i-2]))
            else:
                data[i] = data[i - 1]
    return data


def sanity_check(data):
    """This function checks if the time-series CPET recordings are long enough
    and if the load is monotonically increasing"""
    sanity = True
    load = data["Load"].astype(int)
    derivative = np.diff(load)
    # check if load decreases during test --> not expected behaviour
    if any(derivative < 0):
        pos = np.where(derivative < 0)[0]
        # if decreases only once, then use part of the curve that is valid (i.e. increasing load)
        if len(pos) == 1:
            data = data.iloc[pos[0] + 1:]
        else:
            # if decreases more than once, then remove the recording
            sanity = False
            data = pd.DataFrame()
    # if the test is too short then remove it from the dataset
    if len(data) < 10:
        sanity = False
        data = pd.DataFrame()

    return data, sanity
