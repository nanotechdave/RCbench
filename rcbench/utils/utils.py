import numpy as np
import pandas as pd
import os
import re
from sklearn.linear_model import LinearRegression, Ridge

def find_specific_txt_files(folder_path: str) -> list:
    """
    Scans the given folder for .txt files that contain 'MemoryCapacity' in their names
    but not 'log'.
    
    Parameters:
    - folder_path: str - The path to the folder to be scanned.
    
    Returns:
    - List of paths to the files that match the criteria.
    """
    # List all files in the directory
    all_files = os.listdir(folder_path)
    
    # Compile a regular expression to match the criteria
    pattern = re.compile(r'MemoryCapacity(?!.*log).*\.txt$', re.IGNORECASE)
    
    # Filter files using the regular expression
    matching_files = [file for file in all_files if pattern.search(file)]
    
    # Prepend the folder path to each filename
    full_paths = [os.path.join(folder_path, file) for file in matching_files]
    
    return full_paths

def train_test_split_time_series(data: np.array, target: np.array, test_size=0.2):
    """ Splits data and target into training and test set."""
    split_index = int(len(data) * (1 - test_size))
    data_train = data[:split_index, :]
    data_test = data[split_index:, :]
    target_train = target[:split_index]
    target_test = target[split_index:]
    return data_train, data_test, target_train, target_test

def linear_regression_predict(states_train: np.array, states_test:np.array, target_train: np.array) -> np.array:
    """ Performs linear regression using the states to match the target.
     Returns the predicted waveform"""
    states_train = np.array(states_train)
    states_test = np.array(states_test)
    target_train = np.array(target_train)
    lr = LinearRegression()
    lr.fit(states_train, target_train)
    return lr.predict(states_test)

def ridge_regression_predict(states_train: np.array, states_test: np.array, target_train: np.array, alpha: float = 1.0) -> np.array:
    """Performs ridge regression using the states to match the target.
    Returns the predicted waveform."""
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(states_train, target_train)
    return ridge_reg.predict(states_test)

def extract_voltage_matrix(df: pd.DataFrame) -> np.array:
    """
    Extracts voltage measurements from the DataFrame and creates a matrix of voltages.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing the measurement data.
    
    Returns:
    - np.array - A 2D numpy array (matrix) containing the voltage measurements.
    """
    # Filter columns that contain '_V[' in their column name, indicating a voltage measurement
    voltage_columns = [col for col in df.columns if '_V[' in col]
    
    # Select only the voltage columns from the DataFrame
    voltage_df = df[voltage_columns]
    
    # Convert the DataFrame to a numpy array (matrix)
    voltage_matrix = voltage_df.to_numpy()
    
    return voltage_matrix

def read_and_parse_to_df(filename: str, bias_electrode: str = '08', gnd_electrode: str = '17'):
    # Read the file into a pandas DataFrame
    assert len(bias_electrode)==2 and len(gnd_electrode) ==2 and int(bias_electrode)>0 and int(bias_electrode)<64, "bias_electrode and gnd_electrode must be 2-digit numbers between 01 and 64"
    df = pd.read_csv(filename, sep=r'\s+')
    for col in df.columns:
        df.rename(columns={col: reformat_measurement_header(col)}, inplace=True)
    
    elec_dict = {}
    elec_dict["bias"] = [reformat_measurement_header(str(bias_electrode))]
    elec_dict["gnd"] = [reformat_measurement_header(str(gnd_electrode))]
    elec_dict["float"] = [col.split("_",1)[0] for col in df.columns if isFloat(col, bias_electrode, gnd_electrode)]
    return df, elec_dict

def isFloat(col:str, bias:str, gnd:str) -> bool:
    return ((bias not in col) and (gnd not in col) and "Time" not in col and "I" not in col)

def fillVoltageMatFromDf(measurement:pd.DataFrame, elec_dict:dict) -> list[np.array]:
    bias_voltage = []
    gnd_voltage = []
    float_voltage = []
    
    for col in measurement.columns:
        if any((str(elec) in col and "V" in col) for elec in elec_dict["float"]):
            float_voltage.append(measurement[col].values)
        elif any((str(elec) in col and "V" in col) for elec in elec_dict["bias"]):
            bias_voltage.append(measurement[col].values)
        elif any((str(elec) in col and "V" in col) for elec in elec_dict["gnd"]):
            gnd_voltage.append(measurement[col].values)

    return np.array(bias_voltage).T, np.array(gnd_voltage).T, np.array(float_voltage).T

def reformat_measurement_header(s:str) -> str:
    # Check if the string starts with a single digit
    if len(s) > 0 and s[0].isdigit() and (len(s) == 1 or not s[1].isdigit()):
        # Prefix the string with '0' if it starts with a single digit
        return '0' + s
    else:
        # Return the original string if it doesn't start with a single digit
        return s