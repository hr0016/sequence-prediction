import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from scipy.interpolate import pchip_interpolate
import time
import datetime
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


def eng2ger(val):
    '''
    Translate English column headers to German.
    This function's use case is being able to get a list of German column
    headers for "fields" (to be passed to load_from_csv()), by specifying the
    columns of interest in English. It saves having to compare the lists and
    match up the words manually.
    
    Parameters
    ----------
    val (type: str)
        An English column name to be translated to German.
    
    
    Returns
    -------
    type: str
        German equivalent of the English string.    
    
    '''
    
    # German column headers
    german = np.array(["Schritt","Zustand","Zeit","Programmdauer","Schrittdauer","Zyklus",
                          "Zyklusebene","Prozedur","Prozedurebene","AhAkku","AhLad","AhEla",
                          "AhStep","Energie","WhStep","Spannung","Strom","Temp13"])

    # English translations
    english = np.array(["step", "state", "time", "programme duration", "step duration", "cycle",
                           "cycle level", "procedure", "procedure level", "Qacc", "Qcha", "Qdch",
                           "AhStep", "energy", "WhStep", "voltage", "current", "temp13"])
    
    # First, check if val is in english array
    if val not in english:
        print(f"No translation found for '{val}'")
        return None
    
    else:
        # Find the index of the input string in the English names list
        val_idx = np.where(english == val)[0][0]
        return str(german[val_idx])


def load_from_csv_TEMPERATURE_FIX(fpath, converter, fields=None, translate=None):
    '''
    Handle the problem of the temperature column header changing between files.
    Load the data without specifying which columns to load, then find the temperature
    data and change its name to a generic "temp", so it can be referenced for all files.
            
    We still pass the fields we want (minus temperature) as the "fields" argument,
    but this is no longer used within the pd.read_csv() method.
    
    '''

    # Load from file, without specifying which columns to use
    df = pd.read_csv(fpath, skiprows=[1], header=0, infer_datetime_format=True)
    
    # Get the column name for temperature
    temperature_col_name = [c for c in df.columns if "temp" in c.lower()]
    
    # Append this temperature to the list of field names, in the function scope
    func_fields = fields.copy()
    func_fields.append(temperature_col_name[0])
    
    # Take a subset of the DataFrame containing only the specified fields
    df = df[func_fields]
    
    # Rename the temperature column to a generic name
    df = df.rename(columns={temperature_col_name[0]: "temp"})
        
    # Initialise variable name so there's something to return irrespective of translation bool state
    translation_error_log = None
    
    # Go through translation routine if required. Check for failed translations.
    if translate != None:
        # Translate German columns where a translation exists in the dictionary, else leave the German.
        translated_cols = [translate[ger_col] if ger_col in translate.keys() else ger_col for ger_col in df.columns]
    
        # Store any column names that haven't been translated
        failed_translations = np.where([col not in translate.values() for col in translated_cols])[0]
    
        # This condition fails if len(failed_translations)==0
        if np.any(failed_translations):
            # Add the file path, as well as all failed translations and their column index
            translation_error_log = [fpath, [(idx, df.columns[idx]) for idx in failed_translations]]
    
        # Replace the column names with the translated names
        df.columns = translated_cols
    
    
    # Get rid of state=="STO" and step==9999, since these are useless
    df = df[df.state != "STO"]
    df = df[df.step != 9999]
    
    
    # Get rid of null rows, if present
    df.dropna(inplace=True)
    # Reset the indices in case of null row deletion
    df.reset_index(inplace=True, drop=True)

    return df, translation_error_log


def load_from_csv_EDIT(fpath, converter, fields=None, translate=None): 
    '''
    Load data from CSV files containing Baumhofer 2014 cycle data.
    Handle data type conversion and optionally load a specific set of columns and translate column headers.
    
    Inputs:
        fpath (type: str)
            Path to the CSV file you want to load
        
        converter (type: dict)
            Dict to map data in columns to desired data types.
            Keys: German column names.
            Values: Data type of column
        
        fields (type: list)
            A list of German column names that should be loaded from the CSV file
            
        translate (type: dict)
            A dictionary used to translate the German column headers into English
            
            
    Returns:
        df (type: pd.DataFrame)
            A DataFrame containing the data loaded from the specified CSV file
            
        translation_error_log (type: list)
            A list containing the file path and column indices for any failed translations
        
    '''
    
    # Specify a custom date parser to be used with pd.read_csv
    #custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    
    
    if fields != None:
        df = pd.read_csv(fpath, skiprows=[1], usecols=fields, header=0, dtype=converter, infer_datetime_format=True)
                         #parse_dates=["Zeit"], date_parser=custom_date_parser)
    else:
        df = pd.read_csv(fpath, skiprows=[1], header=0, dtype=converter,  infer_datetime_format=True)
                         #parse_dates=["Zeit"], date_parser=custom_date_parser)
    
    # Initialise variable name so there's something to return irrespective of translation bool state
    translation_error_log = None
    
    # Go through translation routine if required. Check for failed translations.
    if translate != None:
        # Translate German columns where a translation exists in the dictionary, else leave the German.
        translated_cols = [translate[ger_col] if ger_col in translate.keys() else ger_col for ger_col in df.columns]
        
        # Store any column names that haven't been translated
        failed_translations = np.where([col not in translate.values() for col in translated_cols])[0]
        
        # This condition fails if len(failed_translations)==0
        if np.any(failed_translations):
            # Add the file path, as well as all failed translations and their column index
            translation_error_log = [fpath, [(idx, df.columns[idx]) for idx in failed_translations]]
        
        # Replace the column names with the translated names
        df.columns = translated_cols
        
    
    # Get rid of state=="STO" and step==9999, since these are useless
    df = df[df.state != "STO"]
    df = df[df.step != 9999]
    
    
    # Get rid of null rows, if present
    df.dropna(inplace=True)
    # Reset the indices in case of null row deletion
    df.reset_index(inplace=True, drop=True)
    
    return df, translation_error_log


def remove_close_cycles(input_dict, min_val=15, to_plot=False, verbose=False):
    '''
    For the first column of numpy arrays stored as values in input_dict, if there are
    two values closer together than min_val, delete the row corresponding to the
    first of these values.
    
    Repeat for all dictionary keys and return a new dictionary with updated arrays.
    
    
    Parameters
    ----------
    input_dict (type: dict)
    
    
    to_plot (type: bool)
    
    
    verbose (type: bool)
    
       
    
    Returns
    -------
    cap_dict (type: dict)
    
    
    
    '''
    
    # Make a deep copy of the input_dict to avoid altering the original data
    cap_dict = copy.deepcopy(input_dict)    
    
    if to_plot:
        # This is hard-coded, so can introduce problems.
        # In this specific case, I know there should be 8 cells that meet the criteria, so 9 subplots is sufficient.
        fig, ax = plt.subplots(3, 3)
        i = 0

    for k in cap_dict.keys():
        # Extract the cycle numbers and capacity values to make future reference succinct
        cycles = cap_dict[k][:,0]
        caps = cap_dict[k][:,1]

        # Check if there are any cycle numbers closer together than min_val cycles (arbitrarily selected)
        if np.min(np.diff(cycles)) < min_val:
            idx = np.argmin(np.diff(cycles)) + 1
            # Print the information if requested
            if verbose:
                print(f"Dictionary key: {k}")
                print(f"Array index: {idx}")
                print(f"Cycle numbers: {cycles[idx-1:idx+1]}")
                print()

            if to_plot:
                # Plot the original data, including the two closely spaced data points
                ax.flatten()[i].scatter(cycles, caps, marker='o', s=35, color='red')

            # Delete the first of the two closely spaced data points
            cap_dict[k] = np.delete(cap_dict[k], idx-1, axis=0)

            if to_plot:
                # Plot the edited data, excluding the two closely spaced data points
                ax.flatten()[i].scatter(cap_dict[k][:,0], cap_dict[k][:,1], marker='x', s=25, color='cyan')

                # Increment subplot index
                i += 1
    
    if to_plot:
        plt.show()
        
    return cap_dict


def split_adjacent(lst):
    '''
    A simple function to split indices into a nested list.
    Non-consecutive indices are stored in single-element lists
    and consecutive indices are grouped into lists.
    
    For example:
    lst = [1, 3, 4, 6, 8, 10]
    
    split_adjacent(lst)
    >> [[1],
        [3, 4],
        [6],
        [8],
        [10]]
    
    '''
    res = [[lst[0]]]    # start/init with the 1st item/number
    for i in range(1, len(lst)):
        if lst[i] - res[-1][-1] > 1:  # compare current and previous item
            res.append([])
        res[-1].append(lst[i])
    return res


def get_file_index_list(cell_ID, files):
    '''
    Given a list of file paths, identify the cycling and characterisation test files
    that belong to that cell. Additionally, handle the case of consecutive cycling data
    files so that the cycle numbers in between characterisation tests can be continued
    from the first file into the subsequent files without restarting at zero for each
    subsequent file.
    
    Return a list of lists, a example of which looks like this:
    
    [[1]
     [2, 3]
     [4]
     [5]
     ...
     [20]]
     
     Here, the files at indices 2 and 3 are consecutive cycling data files.
     The list of lists returned from the function is used to concatenate
     DataFrames so all the data from one "cycling period" are in one DataFrame.
    
    
    
    
    '''
    # Get the paths for the cycling and characterisation files for cell_ID
    cell_files = np.array([file for file in files if "sanyo "+cell_ID in file])
    # Get the indices of only the cycling files within the list of files for that cell
    cycle_file_indices = np.where(["TBA_Zyk" in fname and "TBA_BOL" not in fname for fname in cell_files])[0]

    output = split_adjacent(cycle_file_indices)

    return output, cell_files


def get_capacity_array_cycle_numbers(cycle_file_indices, cell_filepaths, fields, converter, translate):
    '''
    A function to get the number of cycles elapsed between characterisation tests
    for a particular cell.
    
    Context: The output of this function, along with the capacity values, is used to
             get a continuous array of capacity values for every cycle through
             PCHIP interpolation.
    
    Note: This function could probably be modified to generate the X data (time series)
          too, but I am keeping its responsibility minimal. This does result in inefficiency.
    
    
    Parameters
    ----------
    
    Both of these inputs are generated using the function "get_file_index_list".
    See its code for further information.
    
    cycle_file_indices (type: list)
        A list of lists, containing the indices of the cycling files with respect to the
        "cell_filepaths" list. If there are multiple cycling files present between characterisation
        files, the inner list will contain more than one element, e.g.:
        
            [[1]
             [3, 4]
             [6]
             [8]
             [10]
             ...]
    
    cell_filepaths (type: list)
        A list of all filepaths for a cell.
    
    Returns
    -------
    
    y_arr (type: numpy array)
        Array of cycle numbers corresponding to the cycles at which the
        characterisation tests were performed i.e. the cycle numbers that
        correspond to the capacity measurements.
        
    
    '''
    
    y_arr = np.zeros(shape=(len(cycle_file_indices)+1), dtype=float)

    # Iterate over the cycling periods in between characterisation tests
    # and take care of the case where there are multiple consecutive cycling files.
    for i, indices in enumerate(cycle_file_indices):

        # Create a DataFrame by loading data from the first filepath for the cycling period
        df, _ = load_from_csv_EDIT(cell_filepaths[indices[0]], fields=fields, converter=converter, translate=translate)

        # If there is more than one file for this cycling period,
        # load the remaining ones and concatenate the resulting DataFrames
        if len(indices) > 1:
            for idx in indices[1:]:
                # Load the next CSV file into a temporary DataFrame and concatenate to the existing one
                temp_df, _ = load_from_csv_EDIT(cell_filepaths[idx], fields=fields, converter=converter, translate=translate)
                                
                # Find the maximum value of the cycle column
                max_cycle_val = np.max(df.cycle)
                # Add this max cycle value onto the cycle numbers in the temporary DataFrame to continue the ascending numbers
                temp_df.cycle += max_cycle_val
                # Concatenate DataFrames
                df = pd.concat((df, temp_df), axis=0)

            # Add the maximum cycle value found in df to the appropriate y_arr row
            max_cycle_val = np.max(df.cycle)
            y_arr[i+1] = y_arr[i] + max_cycle_val

        else:
            # Find the maximum value of cycle number
            max_cycle_val = np.max(df.cycle)
            # Add the maximum cycle value found in df to the appropriate y_arr row
            y_arr[i+1] = y_arr[i] + max_cycle_val
                
    return y_arr


def get_capacity_dictionaries(parent_dict, files, fields, converter, translate):
    '''
    Description
    -----------
    
    
    
    
    Parameters
    ----------
    parent_dict (type: dict)
        A dictionary whose values are 2D arrays of cycle_number / capacity.
        
    
    
    Returns
    -------
    y_cap_dict (type: dict)
        A dictionary whose keys are the same as parent_dict.
        The capacity values are the same as those from parent_dict, but
        the cycle numbers are inferred from the data in the CSV cycling files.
    
    
    y_cap_interp_dict (type: dict)
        The same as y_cap_dict, but with a 2D array of interpolated 
        capacity values (obtained using PCHIP), for integer cycle numbers,
        for every cell (key) in the dictionary.        
    
    
    '''



    # Create a dictionary to store the capacity values with the cycle numbers inferred from the CSV files, for each cell
    y_cap_dict = dict.fromkeys(parent_dict)
    y_cap_interp_dict = dict.fromkeys(parent_dict)

    start_time = time.time()
    for cell_ID in parent_dict:
        clear_output(wait=True)
        print(f"Processing {cell_ID}...")

        # For the cell in focus, get the list of filepaths and the indices for the cycling files within that list
        cycle_file_indices, cell_filepaths = get_file_index_list(cell_ID, files)

        # Create a 2D array that will store cycle numbers and capacity values
        y_arr = np.zeros(shape=(len(cycle_file_indices) + 1, 2))

        # Populate the array with capacity values from the dictionary
        y_arr[:, 1] = parent_dict[cell_ID][:,1]

        # Get the cycle numbers corresponding to the capacity measurements, based on the CSV files
        y_arr[:, 0] = get_capacity_array_cycle_numbers(cycle_file_indices=cycle_file_indices,
                                                       cell_filepaths=cell_filepaths,
                                                       fields=fields,
                                                       converter=converter,
                                                       translate=translate)

        # Make the capacity cycle numbers start at 1 so the interpolation result matches the time series cycle dimensions
        y_arr[0,0] = 1

        # Populate the y_cap_dict for the cell in focus
        y_cap_dict[cell_ID] = y_arr

        # Do the PCHIP interpolation to get a continuous array of cycle/capacity values
        x_observed = y_cap_dict[cell_ID][:,0]
        y_observed = y_cap_dict[cell_ID][:,1]
        x_cont = np.arange(x_observed[0], x_observed[-1] + 1)

        # Populate the y_cap_interp_dict with interpolated capacity values for the cell in focus
        y_interp = pchip_interpolate(x_observed, y_observed, x_cont)
        # Stack the cycle numbers and interpolated capacity values into a 2D array and assign to dict key
        y_cap_interp_dict[cell_ID] = np.vstack((x_cont, y_interp)).T


    print("Finished.")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    
    return y_cap_dict, y_cap_interp_dict


def find_empty_time_series_arrays(result_dict):
    # Create a list to hold the cell ID and cycle number of any empty time series cycle data
    empty = []
    cells = 0
    cycles = 0
    
    for cell in result_dict:
        # Increment counter
        cells += 1
        
        for cycle in result_dict[cell]['ts_data']:
            # Increment counter
            cycles += 1
            
            # Extract the time data array
            time_arr = result_dict[cell]['ts_data'][cycle]['t'].copy()

            # There are some time series arrays that have zero elements when the charge state and voltage bounds are imposed.
            # Check for this and skip them if that's true. Handle it in a better way later.
            if time_arr.shape[0] == 0:
                empty.append([cell, cycle])
    
    print(f"Checked {cells} cells, {cycles} cycles.")
    
    if len(empty) == 0:
        print("No empty time series arrays found")
    else:
        print("Empty time series arrays found")
    
    return empty


def apply_array_length_threshold(cell_dict, threshold_val, verbose=False):
    '''
    
    Parameters
    ----------
    cell_dict (type: dict)
    
    
    threshold_val (type: int)
        
    
    '''
    
    # Consider a lower threshold for the number of samples in a voltage array
    # Any cycles that have fewer samples than the threshold should be discounted

    # Make a copy of the dictionary passed as input, to avoid altering the original
    c_dict = copy.deepcopy(cell_dict)
    
    # Create a dictionary that contains the number of time series data samples present in each cycle
    c_dict['cycle_ts_sample_count'] = {cycle: len(c_dict['ts_data'][cycle]['V']) for cycle in c_dict['ts_cycles']}
    
    # Identify the cycles that should be kept, based on the threshold_val
    c_dict["ts_cycles_to_keep"] = np.array([k for k,v in c_dict['cycle_ts_sample_count'].items() if v > threshold_val])

    if verbose:
        print(f"Threshold of {threshold_val} samples: Keep {100 * len(c_dict['ts_cycles_to_keep']) / len(c_dict['ts_cycles']):.2f}% of cycles")
        
    # Store the time series data for the cycles that remain after thresholding    
    c_dict['ts_data_thresh'] = {cycle: c_dict['ts_data'][cycle].copy() for cycle in c_dict['ts_cycles_to_keep']}


    # Construct a capacity array that contains only the cycles that remain after thresholding
    cap_arr = []
    for cycle in c_dict['ts_cycles_to_keep']:
        idx = np.where(c_dict['capacity'][:,0]==cycle)[0]
        cap_arr.append(c_dict['capacity'][idx, :])
    
    # Turn it into a numpy array with the correct dimensions
    cap_arr = np.array(cap_arr)
    cap_arr = cap_arr.squeeze()
    
    print(cap_arr.shape)

    # Assign the thresholded capacity array to a new dict key
    c_dict['capacity_thresh'] = cap_arr

    
    # Debugging statements to make sure that the cycle numbers match
    assert(np.all(c_dict['capacity_thresh'][:,0] == c_dict['ts_cycles_to_keep']))
    assert(np.all(list(c_dict['ts_data_thresh'].keys()) == c_dict['ts_cycles_to_keep']))
    
    return c_dict


def pad_time_series_data(parent_dict, mask_val=np.nan, verbose=False):
    '''
    Description.


    Parameters
    ----------

    parent_dict (type: dict)
        Description.

    mask_val (type: float)
        Value to use when extending arrays. Defaults to np.nan so it doesn't affect scaling/normalisation.


    '''
    if mask_val == None:
        mask_val = np.nan

    # Make a copy of the dictionary passed to the function to avoid manipulating the original data
    new_dict = copy.deepcopy(parent_dict)

    # Initialise a variable for storing the maximum array length
    max_len = 0

    # Iterative method to find max time series array length
    for cell in new_dict:
        for cycle in new_dict[cell]['ts_data_thresh']:
            # Get the value of the first time series data key e.g. 'V' for voltage.
            # Assumes all variable arrays are the same length
            k = list(new_dict[cell]['ts_data_thresh'][cycle].keys())[0]

            # Find the length of the array for that cycle
            arr_len = new_dict[cell]['ts_data_thresh'][cycle][k].shape[0]
            if arr_len > max_len:
                max_len = arr_len   

    if verbose: print(max_len)

    # Create a new sub-dictionary to store the padded result
    for cell in new_dict.keys():
        new_dict[cell]['ts_padded'] = {cycle:
                                         {var: None for var in new_dict[cell]['ts_data_thresh'][cycle].keys()}
                                     for cycle in new_dict[cell]['ts_data_thresh'].keys()}


    # Loop through the time series arrays and extend them with np.nan, to have length of max_len
    # For each cycle
    for cell in new_dict:
        for cycle in new_dict[cell]['ts_data_thresh'].keys():
            # For each variable e.g. voltage, time
            for var in new_dict[cell]['ts_data_thresh'][cycle]:
                # Store the data in a variable for readability
                arr = new_dict[cell]['ts_data_thresh'][cycle][var]
                # Find the current length of the array
                current_len = arr.shape[0]
                # Specify how many padded values to add
                num_pad_vals_to_add = max_len - current_len

                # Generate the padded array
                padded_arr = np.append(arr, np.repeat(mask_val, num_pad_vals_to_add))
                # Assign it to the relevant location in padded_cell_dict
                new_dict[cell]['ts_padded'][cycle][var] = padded_arr


    return new_dict


def pad_time_series_data_EDIT(parent_dict, key, mask_val=np.nan, verbose=False):
    '''
    Description.


    Parameters
    ----------

    parent_dict (type: dict)
        Description.
        
    key (type: str)
        The dictionary key that specifies one level above the "cycle" level.
        This makes it more flexible, slightly lowering the dependence on the 
        specification of the input dictionary

    mask_val (type: float)
        Value to use when extending arrays. Defaults to np.nan so it doesn't affect scaling/normalisation.


    '''
    if mask_val == None:
        mask_val = np.nan

    # Make a copy of the dictionary passed to the function to avoid manipulating the original data
    new_dict = copy.deepcopy(parent_dict)

    # Initialise a variable for storing the maximum array length
    max_len = 0

    # Iterative method to find max time series array length
    for cell in new_dict:
        for cycle in new_dict[cell][key]:
            # Get the value of the first time series data key e.g. 'V' for voltage.
            # Assumes all variable arrays are the same length
            k = list(new_dict[cell][key][cycle].keys())[0]

            # Find the length of the array for that cycle
            arr_len = new_dict[cell][key][cycle][k].shape[0]
            if arr_len > max_len:
                max_len = arr_len   

    if verbose: print(max_len)

    # Create a new sub-dictionary to store the padded result
    for cell in new_dict.keys():
        new_dict[cell]['ts_padded'] = {cycle:
                                         {var: None for var in new_dict[cell][key][cycle].keys()}
                                     for cycle in new_dict[cell][key].keys()}


    # Loop through the time series arrays and extend them with np.nan, to have length of max_len
    # For each cycle
    for cell in tqdm(list(new_dict.keys())):
        for cycle in new_dict[cell][key].keys():
            # For each variable e.g. voltage, time
            for var in new_dict[cell][key][cycle]:
                # Store the data in a variable for readability
                arr = new_dict[cell][key][cycle][var]
                # Find the current length of the array
                current_len = arr.shape[0]
                # Specify how many padded values to add
                num_pad_vals_to_add = max_len - current_len

                # Generate the padded array
                padded_arr = np.append(arr, np.repeat(mask_val, num_pad_vals_to_add))
                # Assign it to the relevant location in padded_cell_dict
                new_dict[cell]['ts_padded'][cycle][var] = padded_arr


    return new_dict



def construct_3d_x_array(input_dict, variables=['V', 't_elapsed'], key='ts_padded'):
    '''
    
    The resulting array has the shape [num_features, num_cycles, num_samples],
    where:
        num_features refers to voltage and elapsed cycle time (seconds),
        num_cycles refers to the number of cycles over all cells,
        num_samples refers to the number of measurements taken within a cycle
    
    
    Parameters
    ----------
    input_dict (type: dict)
        Description
        
    
    variables (type: list)
        Description    
    
    
    Returns
    -------
    X_arr (type: numpy array)
        Description
        
    
    cell_cycle_indices (type: numpy array)
        Description
    
    
    
    '''
    
    
    
    # Get a list of cells from the input dict keys
    cells = list(input_dict.keys())
    
    # Initialise the 3D array using the padded time series data from the first cell
    X_arr = np.array([[input_dict[cells[0]][key][cycle][var] for cycle in input_dict[cells[0]][key].keys()] for var in variables])
    
    # Initialise a 1D array that will tell us which cell the cycle came from
    #cell_cycle_indices = np.array([str(cells[0]) for i in range(X_arr.shape[1])])
    cell_cycle_indices = np.array([str(cells[0])+"_"+str(k) for k in input_dict[cells[0]][key].keys()])
    
    # Build up the complete arrays by creating the arrays per cell, then appending them to the master arrays
    for cell in cells[1:]:
        temp_arr = np.array([[input_dict[cell][key][cycle][var] for cycle in input_dict[cell][key].keys()] for var in variables])
        X_arr = np.append(X_arr, temp_arr, axis=1)

        #temp_indices_arr = np.array([str(cell) for i in range(temp_arr.shape[1])])
        temp_indices_arr = np.array([str(cell)+"_"+str(k)  for k in input_dict[cell][key].keys()])
        cell_cycle_indices = np.append(cell_cycle_indices, temp_indices_arr)
        
        
    return X_arr, cell_cycle_indices


def construct_y_soh_array(input_dict):
    '''
    Specifically consider the SOH (remaining capacity) for the case of the
    Baumhofer method.
    
    
    
    '''

    # Get a list of cells from the input dict keys
    cells = list(input_dict.keys())
    
    # Initialise the y array using the capacity data from the first cell
    y_arr = np.array([input_dict[cells[0]]['capacity_thresh'][:,1]])
    
    # Initialise a 1D array that will tell us which cell the cycle came from
    #cell_cycle_indices = np.array([str(cells[0]) for i in range(X_arr.shape[1])])
    cell_cycle_indices = np.array([str(cells[0])+"_"+str(int(k)) for k in input_dict[cells[0]]['capacity_thresh'][:,0]])
    
    # Build up the complete arrays by creating the arrays per cell, then appending them to the master arrays
    for cell in cells[1:]:
        temp_arr = np.array([input_dict[cell]['capacity_thresh'][:,1]])
        y_arr = np.append(y_arr, temp_arr, axis=1)

        temp_indices_arr = np.array([str(cell)+"_"+str(int(k)) for k in input_dict[cell]['capacity_thresh'][:,0]])
        cell_cycle_indices = np.append(cell_cycle_indices, temp_indices_arr)
        
        
    return y_arr, cell_cycle_indices


def get_train_val_test(cell_list, X, y, index, split_frac_1=0.7, split_frac_2=0.5, rdm_state=None):
    '''
    Given a list of cell IDs (cell_list), split the cell IDs into train, validation and test lists.
    Then, using index, the relevant rows of data in X are located for train, validation and test cells.
    These subsets are then extracted and returned.
    
    
    Parameters
    ----------
    cell_list (type: list)
        A list of cell IDs, from which the data in X are derived, to be distributed among
        training, validation and testing subsets.
        
    X (type: numpy array)
        3D array of data. Shape: [num_features, num_cycles, num_samples].
        
    y (type: numpy array)
        Description...
        
    index (type: numpy array)
        Array with the same shape as the number of instances in the input X.
        Each element has the format "cell_cycle" e.g. "002_1029".
        This array is used to identify the rows of X belonging to particular cells.
        
    split_frac_1 (type: float)
        The fraction of the cells in cell_list to be used for training cells.
        
    split_frac_2 (type: float)
        The fraction used to split the remaining cells after training cells have been selected.
        For example if split_frac_1 == 0.7, then we have 30% of the cells in cell_list left.
        Setting split_frac_2 to be 0.5 gives us 15% of the cells in cell_list for each of validation and testing.
    
    rdm_state (type: int)
        The random state to be passed to sklearn train_test_split()
    
    
    Returns
    -------
    X_train (type: numpy array)
        3D array of X data from training cells
    
    X_val (type: numpy array)
        3D array of X data from validation cells
    
    X_test (type: numpy array)
        3D array of X data from testing cells
    
    '''
    
    # Input validation
    if split_frac_1 == None:
        split_frac_1 = 0.7
    if split_frac_2 == None:
        split_frac_2 = 0.5
        
    # Get cell IDs for train, validation and test sets
    train_cells, the_rest = train_test_split(cell_list, train_size=split_frac_1, random_state=rdm_state)
    val_cells, test_cells = train_test_split(the_rest, train_size=split_frac_2, random_state=rdm_state)

    # Create arrays of boolean values that tell us, for each index, whether or not that data is in a particular set.
    # The array called index contains elements with the form "003_1039" i.e. "cell_cycle".
    # Inside the list comprehension we consider only the first 3 characters of the string, to isolate the cell ID.
    train_bool = np.array([idx[0:3] in train_cells for idx in index])
    val_bool = np.array([idx[0:3] in val_cells for idx in index])
    test_bool = np.array([idx[0:3] in test_cells for idx in index])

    assert(np.all(train_bool + test_bool + val_bool) == 1)

    # Use these boolean arrays to take subsets of the X array
    X_train = X[:, train_bool, :]
    X_val = X[:, val_bool, :]
    X_test = X[:, test_bool, :]
    
    # Use these boolean arrays to take subsets of the y array
    y_train = y[train_bool, ...]
    y_val = y[val_bool, ...]
    y_test = y[test_bool, ...]
    
    # Create a dictionary that tells us which cells were used for train/val/test
    cells_dict = {'train': train_cells, 'val': val_cells, 'test': test_cells}
    
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), cells_dict


def min_max_scaling(X_train, X_val, X_test):
    '''
    OBSOLETE FUNCTION. REPLACED BY scaler_3d().
    
    Given data for train, val and test, compute the minimum and maximum values (per-feature)
    in the training set and scale the train, val and test sets using min max scaling with the
    min and max values obtained from the training set.    
    
    Parameters
    ----------
    X_train (type: numpy array)
        3D array of training data. Shape: [num_features, num_cycles, num_samples].
    
    X_val (type: numpy array)
        3D array of validation data. Shape: [num_features, num_cycles, num_samples].
        
    X_test (type: numpy array)
        3D array of test data. Shape: [num_features, num_cycles, num_samples].
    
    
    Returns
    -------
    X_train_sc (type: numpy array)
        3D array of training data scaled per-feature using the minimum and maximum values in X_train.
        Shape: [num_features, num_cycles, num_samples].
    
    X_val_sc (type: numpy array)
        3D array of training data scaled per-feature using the minimum and maximum values in X_train.
        Shape: [num_features, num_cycles, num_samples].
    
    X_test_sc (type: numpy array)
        3D array of training data scaled per-feature using the minimum and maximum values in X_train.
        Shape: [num_features, num_cycles, num_samples].
        
    '''
    
    # Obtain the max and min values for each feature, from the training data, to use for scaling
    max_vals = np.array([np.nanmax(X_train[i, :, :]) for i in range(X_train.shape[0])])
    min_vals = np.array([np.nanmin(X_train[i, :, :]) for i in range(X_train.shape[0])])

    # Generate the arrays to hold the scaled values
    X_train_sc = np.copy(X_train)
    X_val_sc = np.copy(X_val)
    X_test_sc = np.copy(X_test)

    # Do the min max scaling per feature (voltage, time)
    for i in range(len(min_vals)):
        X_train_sc[i, ...] = (X_train_sc[i,...] - min_vals[i]) / (max_vals[i] - min_vals[i])
        X_val_sc[i, ...] = (X_val_sc[i,...] - min_vals[i]) / (max_vals[i] - min_vals[i])
        X_test_sc[i, ...] = (X_test_sc[i,...] - min_vals[i]) / (max_vals[i] - min_vals[i])

    # Convert nan values to zeros, if present
    X_train_sc = np.nan_to_num(X_train_sc, nan=0)
    X_val_sc = np.nan_to_num(X_val_sc, nan=0)
    X_test_sc = np.nan_to_num(X_test_sc, nan=0)
    
    
    return X_train_sc, X_val_sc, X_test_sc


def scaler_3d(X_train, X_val, X_test, scaler_type='robust', return_scaler=False):
    
    # Set return_scaler = True if we want to return the scaler
    
    # Instantiate the selected scaler type
    if scaler_type.lower() == "robust":
        sc = RobustScaler()
    elif scaler_type.lower() == "standard":
        sc = StandardScaler()
    elif scaler_type.lower() == "minmax":
        sc = MinMaxScaler()
    else:
        print("Invalid scaler_type argument")
        return None, None, None
    
    
    # Assign meaning to the shape of the input X array, for readability
    num_instances, num_steps, num_features = X_train.shape
    
    # Fit the scaler using the training data and transform the training data
    temp = X_train.reshape(-1, num_features)
    temp_sc = sc.fit_transform(temp)
    X_train_sc = temp_sc.reshape(-1, num_steps, num_features)

    # Use the scaler (fit on the training data), to transform the validation data
    temp = X_val.reshape(-1, num_features)
    temp_sc = sc.transform(temp)
    X_val_sc = temp_sc.reshape(-1, num_steps, num_features)

    # Use the scaler (fit on the training data), to transform the testing data
    temp = X_test.reshape(-1, num_features)
    temp_sc = sc.transform(temp)
    X_test_sc = temp_sc.reshape(-1, num_steps, num_features)
    
    
    # Convert nan values to zeros, if present
    X_train_sc = np.nan_to_num(X_train_sc, nan=0)
    X_val_sc = np.nan_to_num(X_val_sc, nan=0)
    X_test_sc = np.nan_to_num(X_test_sc, nan=0)
    
    
    if return_scaler:
        return X_train_sc, X_val_sc, X_test_sc, sc
    else:
        return X_train_sc, X_val_sc, X_test_sc


def create_y_target_array(parent_dict, cell_ID, df):
    '''
    For a particular cell_ID in parent_dict, generate an array with 5 columns.
    
    These columns represent:
    - Number of cycles remaining until knee onset
    - Number of cycles remaining until knee point
    - Number of cycles remaining until EOL
    - Capacity degradation (Ah) until knee onset
    - Capacity degradation (Ah) until knee point
    
    
    Parameters
    ----------
    parent_dict (type: dict)
        Dictionary whose keys are cell IDs, containing 2D cycle/capacity array in a key called 'capacity_thresh'
        
    cell_ID (type: str)
        Cell identifier string used to specify the keys of parent_dict to extract cell data.
        
    df (type: pd.DataFrame)
        A DataFrame containing, for each cell, 5 values (cycle number for onset, point, EOL, capacity at onset and point).
        This is obtained using the function "get_knee_and_eol_results"    
    
    
    '''
    
    
    # Extract the 2D array of cycles/interpolated capacity values from the dictionary
    cap_arr = copy.deepcopy(parent_dict[cell_ID]['capacity_thresh'])

    # Create a DataFrame so we can explicitly refer to the column names for assignment
    result = pd.DataFrame(np.zeros(shape=(cap_arr.shape[0], 5), dtype=float),
                          index=cap_arr[:,0].astype(int),
                          columns=['tto', 'ttp', 'tte', 'deg_o', 'deg_p'])

    # Populate the result DataFrame with values
    result['tto'] = df.at[cell_ID, "onset"] - cap_arr[:,0]
    result['ttp'] = df.at[cell_ID, "point"] - cap_arr[:,0]
    result['tte'] = df.at[cell_ID, "EOL"] - cap_arr[:,0]
    result['deg_o'] = cap_arr[:,1] - df.at[cell_ID, "onset_y"]
    result['deg_p'] = cap_arr[:,1] - df.at[cell_ID, "point_y"]

    # Convert the DataFrame to a numpy array
    result_arr = result.to_numpy(copy=True)
    
    return result_arr


def interpolate_data(data_dict, variables=['V', 'I', 'T', 'Q'], time_freq=4):
    
    # Initialise dict
    interpld_data = dict()

    for cell in tqdm(list(data_dict.keys())):

        # Initialise cell dict
        interpld_data[cell] = dict()
        interpld_data[cell]['interp'] = dict()

        for cycle in data_dict[cell]['ts_data_thresh']:

            # Initialise cycle dict
            interpld_data[cell]['interp'][cycle] = dict()

            time = data_dict[cell]['ts_data_thresh'][cycle]['t_elapsed']
            current = data_dict[cell]['ts_data_thresh'][cycle]['I']
            voltage = data_dict[cell]['ts_data_thresh'][cycle]['V']
            temperature = data_dict[cell]['ts_data_thresh'][cycle]['T']
            charge = data_dict[cell]['ts_data_thresh'][cycle]['QCharge']

            regular_time = np.arange(0, np.max(time), time_freq)

            if 'I' in variables:
                f_i = interp1d(time, current)
                interpld_data[cell]['interp'][cycle]['I'] = f_i(regular_time)  
            if 'V' in variables:
                f_v = interp1d(time, voltage)
                interpld_data[cell]['interp'][cycle]['V'] = f_v(regular_time)
            if 'T' in variables:
                f_T = interp1d(time, temperature)
                interpld_data[cell]['interp'][cycle]['T'] = f_T(regular_time)
            if 'Q' in variables:
                f_Q = interp1d(time, charge)
                interpld_data[cell]['interp'][cycle]['QCharge'] = f_Q(regular_time)                
    
    return interpld_data


# A function to reshape the data for CNN
def reshape_for_cnn(X_arr, to_plot=False):
    '''
    
    
    
    
    
    
    '''
    # Get the shape of the data prior to reshaping for LSTM
    features, samples, timesteps = X_arr.shape
    
    # Initialise an empty array to hold the reshaped X array
    X_reshaped = np.zeros(shape=(samples, timesteps, features), dtype=float)
    
    for i in range(X_reshaped.shape[0]):
        X_reshaped[i] = np.vstack([X_arr[0, i, :], X_arr[1, i, :], X_arr[2, i, :]]).T
        
    if to_plot:
        # Plot a random selection of instances to check they look OK
        indices = np.random.randint(0, samples, size=25)
        fig, ax = plt.subplots(5,5)
        for subplot, sample in enumerate(indices):
            ax.flatten()[subplot].plot(X_reshaped[sample,:,0], X_reshaped[sample,:,2], 'o')

        plt.show()
        
    return X_reshaped















