import copy
import os
import tqdm
import datetime
from pathlib import Path
from _utils.assert_quality import *


def fetch_data(path, sex_file_path=None):
    """Load the data into two dataframes, one for men and one for women"""
    patients = [file.split("_")[0] for file in os.listdir(path)]
    df = pd.DataFrame()

    dataset = pd.DataFrame({"Patient IDs": patients, "CPET Data": [df] * len(patients)})
    rejected_patients = []
    for pos, file in tqdm.tqdm(enumerate(os.listdir(path)), total=len(patients)):
        df = pd.read_csv(path + "/" + file, delimiter="\t", encoding='unicode_escape')
        df = df.set_axis([i.strip() for i in df.columns], axis=1)
        # separate the two types of .txt files
        try:
            # drop the column of the T-phase measurements in files with the second format
            df.drop(["t-ph", "BR", "HRR"], axis=1, inplace=True)
            # find the lines that indicate the anaerobic thresholds.
            empty_lines = df.loc[df["Load"].str.strip() == ""].index
            # Drop the lines with the label indicating the start of the anaerobic threshold,
            # test and recovery phase and the units of the measurements
            df.drop(0, inplace=True)
            df.drop(empty_lines, inplace=True)
        except KeyError:
            # drop the columns with the T-phase and Afst. meaurements
            df.drop(["Tfase", "Afst.", "VO2slo", "BR (%)", "HRR (L)"], axis=1, inplace=True)
            # find the lines that indicate the anaerobic thresholds.
            threshold_lines = df.loc[pd.isna(df["Belasting"])].index
            # Drop the lines with the label indicating the start of the anaerobic threshold,
            # test and recovery phase and the units of the measurements
            df.drop(0, inplace=True)
            df.drop(threshold_lines, inplace=True)
        df, flag = slice_dataframe(df)
        if flag:
            df.rename(columns={'Tijd': 'Time', 'Belasting': 'Load'}, inplace=True)
            df = convert_time(df)
            df, sanity = sanity_check(df)
            if sanity:
                df = check_for_missing_values(df)
                df = check_for_intermediate_errors(df)
                df = check_for_ending_zeros(df)

                df_filtered = correct_measurements(df)
                df_filtered = filter_quality(df, df_filtered)
                df_filtered = restore_format(df_filtered)
                df_filtered = rearrange_labels(df_filtered)
                dataset.at[pos, "CPET Data"] = df_filtered.reset_index(drop=True)
            else:
                print(f"{file} has been removed due to sanity check")
                rejected_patients.append(pos)
        else:
            print(f"{file} has been removed due to empty data")
            rejected_patients.append(pos)

    dataset.drop(rejected_patients, inplace=True)
    men, women = separate_data_based_on_sex(dataset, sex_file_path)
    return men.reset_index(drop=True), women.reset_index(drop=True)


def slice_dataframe(data):
    """Replaces the invalid characters and keeps i) three samples during resting phase and
    ii) the test phase """
    dataframe = copy.deepcopy(data)
    dataframe = dataframe.reset_index(drop=True)
    non_empty_flag = True
    try:
        # find the index of the maximum achieved load. Indicates the end of test phase
        end_index = dataframe["Load"].astype("float")[::-1].idxmax()
        if pd.isna(dataframe.iloc[end_index]["V'O2"]):
            end_index = dataframe["V'O2"].last_valid_index()

        # if True no VCO2, VO2 values --> return empty dataframe
        if end_index is None:
            non_empty_flag = False
            return pd.DataFrame(), non_empty_flag

        # find the positions of zero load
        # positions before the "end_index" indicate the baseline state
        zero_load_pos = dataframe["Load"].loc[dataframe["Load"].str.strip() == "0"].index
        zero_load_pos = [i for i in zero_load_pos if i < end_index]

        # filter the zero load position. Position indices must be continuous
        diff = np.diff(zero_load_pos) == 1
        index = [i for i, x in enumerate(diff) if not x]
        if index:
            zero_load_pos = zero_load_pos[:index[0]]

    except KeyError:
        # find the index of the maximum achieved load. Indicates the end of test phase
        end_index = dataframe["Belasting"].astype("float")[::-1].idxmax()
        if pd.isna(dataframe.iloc[end_index]["V'O2"]):
            end_index = dataframe["V'O2"].iloc[:end_index].last_valid_index()

            # if True no VCO2, VO2 values --> return empty dataframe
            if end_index is None:
                non_empty_flag = False
                return pd.DataFrame(), non_empty_flag

        # find the positions of zero load
        # positions before the "end_index" indicate the baseline state
        zero_load_pos = dataframe["Belasting"].loc[dataframe["Belasting"].str.strip() == "0"].index
        zero_load_pos = [i for i in zero_load_pos if i < end_index]

        # filter the zero load position. Position indices must be continuous
        diff = np.diff(zero_load_pos) == 1
        index = [i for i, x in enumerate(diff) if not x]
        if index:
            zero_load_pos = zero_load_pos[:index[0]]

    # take at most 3 measurements from baseline
    if len(zero_load_pos) > 3:
        start_index = zero_load_pos[-3]
    else:
        start_index = 0
    # slice the dataframe to the desired indexes
    dataframe = dataframe[start_index:end_index]

    # remove special characters
    dataframe["V'O2"] = dataframe["V'O2"].str.replace("«|-", "0", regex=True)
    dataframe["V'CO2"] = dataframe["V'CO2"].str.replace("«|-", "0", regex=True)
    dataframe["V'E"] = dataframe["V'E"].str.replace("«|-", "0", regex=True)
    dataframe["PECO2"] = dataframe["PECO2"].str.replace("«|-", "0", regex=True)
    dataframe["PEO2"] = dataframe["PEO2"].str.replace("«|-", "0", regex=True)
    dataframe["PETCO2"] = dataframe["PETCO2"].str.replace("«|-", "0", regex=True)
    dataframe["PETO2"] = dataframe["PETO2"].str.replace("«|-", "0", regex=True)
    dataframe["BF"] = dataframe["BF"].str.replace("«|-", "0", regex=True)
    return dataframe, non_empty_flag


def correct_measurements(df):
    """Implements the second step of the pre-processing approach ny applying the moving average filter and/or the
    local statistical approach"""

    dataframe = copy.deepcopy(df)
    labels = ["V'CO2", "V'O2", "BF", "V'E", "PECO2", "PETCO2", "PEO2", "PETO2", "HR"]
    for i in dataframe.columns:
        # fill the missing entries of blood pressure with the first valid measurement
        if i == "Psys" or i == "Pdia":
            try:
                val = dataframe[i].loc[dataframe[i].first_valid_index()]
                dataframe[i] = dataframe[i].fillna(val)
            except KeyError:
                pass

        # use the moving average filter to remove wrong measurements
        elif i in labels and i != "BF":
            dataframe[i] = dataframe[i].fillna("0")
            data = [x.strip() if isinstance(x, str) else x for x in dataframe[i]]
            data = ["0" if d == "-" else d for d in data]
            data = correct_outliers(np.array(data).astype(float), 7)
            if len(dataframe[i]) > 30:
                dataframe[i] = moving_average(np.array(data).astype(float), 11)
            else:
                dataframe[i] = correct_outliers(np.array(data).astype(float), 5)

        elif i == "BF":
            dataframe[i] = dataframe[i].fillna("0")
            data = [x.strip() if isinstance(x, str) else x for x in dataframe[i]]
            data = ["0" if d == "-" else d for d in data]
            data = correct_outliers(np.array(data).astype("float"), 5)
            if len(dataframe[i]) > 30:
                dataframe[i] = moving_average(np.array(data).astype(float), 3)
            else:
                dataframe[i] = correct_outliers(np.array(data).astype(float), 5)
        else:
            continue
    return dataframe


def pad_data(data, window):
    """Function to pad the data. Padding is necessary to retain the dimension
    of the data after the moving average filter"""
    padded_data = [None] * (len(data) + window - 1)
    padded_data[0:int((window - 1) / 2)] = data[int((window - 1) / 2):0:-1]
    padded_data[int((window - 1) / 2):len(data) + int((window - 1) / 2)] = data
    padded_data[len(data) + int((window - 1) / 2): len(padded_data)] = data[-2:-int((window - 1) / 2) - 2:-1]
    return padded_data


def correct_outliers(data, window):
    """Implements the local statistical correction method."""
    padded_data = pad_data(data, window)
    for i in range(int((window - 1) / 2), len(data) + int((window - 1) / 2)):
        # isolate the values around the position to be filtered
        temp = padded_data[i - int((window - 1) / 2):i + int((window - 1) / 2) + 1]

        # remove the value to be filtered, so that not to affect the result
        temp = np.delete(temp, int((len(temp) - 1) / 2))

        avg_value = np.mean(temp)
        std_value = np.std(temp)

        if padded_data[i] > avg_value + std_value or padded_data[i] < avg_value - std_value:
            padded_data[i] = (padded_data[i - 1] + padded_data[i + 1]) / 2
    return padded_data[int((window - 1) / 2): len(data) + int((window - 1) / 2)]


def moving_average(data, window):
    return np.convolve(pad_data(data, window), np.ones(window), 'valid') / window


def convert_time(dataframe):
    """Transforms time into seconds"""
    try:
        date_time = [datetime.datetime.strptime(dtime, " %M:%S") for dtime in dataframe[dataframe.columns[0]]]
    except ValueError:
        date_time = [datetime.datetime.strptime(dtime, " %M:%S ") for dtime in dataframe[dataframe.columns[0]]]
    a_timedelta = [i - datetime.datetime(1900, 1, 1, 0) for i in date_time]
    seconds = [float(i.total_seconds()) for i in a_timedelta]

    # replace the transformed values
    dataframe[dataframe.columns[0]] = seconds
    return dataframe


def restore_format(dataframe):
    """Restores the data-type of the variables"""

    dataframe["V'CO2"] = dataframe["V'CO2"].astype(int)
    dataframe["V'O2"] = dataframe["V'O2"].astype(int)
    dataframe["BF"] = dataframe["BF"].astype(int)
    dataframe["V'E"] = np.round(dataframe["V'E"].astype(float), 2)
    dataframe["PECO2"] = np.round(dataframe["PECO2"].astype(float), 2)
    dataframe["PEO2"] = np.round(dataframe["PEO2"].astype(float), 2)
    dataframe["PETCO2"] = np.round(dataframe["PETCO2"].astype(float), 2)
    dataframe["PETO2"] = np.round(dataframe["PETO2"].astype(float), 2)
    dataframe["Load"] = dataframe["Load"].astype(float)
    dataframe["HR"] = dataframe["HR"].astype(int)
    dataframe["Psys"] = dataframe["Psys"].astype(int)
    dataframe["Pdia"] = dataframe["Pdia"].astype(int)
    return dataframe


def rearrange_labels(dataframe):
    if "Tijd" in dataframe.columns:
        # rearrange the sequence of the columns to match the other .txt format
        col_sequence = ["Tijd", "Belasting", "HR", "Psys", "Pdia", "V'O2", "V'CO2", "BF", "V'E", "RER", "PECO2",
                        "PETCO2", "PEO2", "PETO2"]
        dataframe = dataframe[col_sequence]
    else:
        col_sequence = ["Time", "Load", "HR", "Psys", "Pdia", "V'O2", "V'CO2", "BF", "V'E", "RER", "PECO2",
                        "PETCO2", "PEO2", "PETO2"]
        dataframe = dataframe[col_sequence]
    dataframe.rename(columns={'Tijd': 'Time', 'Belasting': 'Load'}, inplace=True)
    return dataframe


def separate_data_based_on_sex(dataframe, sex_file):
    """Separate the original dataset into two subgroups based on the sex"""

    sex_db = pd.read_excel(sex_file)
    males_id = list(sex_db["record_id"][sex_db["sex"].astype(int) == 0])
    females_id = list(sex_db["record_id"][sex_db["sex"].astype(int) == 1])
    _, a_ind, _ = np.intersect1d(list(dataframe["Patient IDs"]), males_id, return_indices=True)
    m = dataframe.iloc[a_ind]

    _, a_ind, _ = np.intersect1d(list(dataframe["Patient IDs"]), females_id, return_indices=True)
    f = dataframe.iloc[a_ind]
    return m, f


def filter_quality(original, filtered):

    filtered_rer = filtered["V'CO2"] / filtered["V'O2"]
    try:
        orig_rer = original["RER"].astype(float)
    except KeyError:
        orig_rer = original["V'CO2"].astype(float) / original["V'O2"].astype(float)
    # in case of invalid symbol e.g. "-" or ">>"
    except ValueError:
        orig_rer = original["V'CO2"].astype(float) / original["V'O2"].astype(float)

    dist = orig_rer - filtered_rer
    if any(np.abs(dist) > 0.15) and not any(dist.isna()):
        if len(filtered["V'CO2"]) > 30:
            filtered["RER"] = moving_average(data=orig_rer, window=11)
        else:
            filtered["RER"] = moving_average(data=orig_rer, window=5)
    else:
        filtered["RER"] = filtered_rer
    return filtered


if __name__ == "__main__":
    print("Running 'load_data' module as main script.")
    parent_folder = os.path.join(*Path(os.getcwd()).parts[:-2])
    data_path = os.path.join(parent_folder, "Data/iCOMPEER")
    path_to_sex_file = os.path.join(parent_folder, "Data")
    save_data_path = os.path.join(parent_folder, "Results")
    males, females = fetch_data(data_path, path_to_sex_file)
    print("Finished")
