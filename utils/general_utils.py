
import re
import streamlit as st
import pandas as pd
import json

### coordinate parser functions
def wafer_structure_to_dataframe(wafer_structure):
    """
    Converts the wafer structure dictionary into a Pandas DataFrame.

    Args:
        wafer_structure (dict): A dictionary where keys are die numbers and values are dictionaries 
                                with block numbers and their coordinates.

    Returns:
        pd.DataFrame: A DataFrame containing the die, block, x, and y coordinates.

    Example:
        df = wafer_structure_to_dataframe(wafer_structure)
    """
    rows = []
    for die_num, blocks in wafer_structure.items():
        for block_num, coordinates in blocks.items():
            for coordinate in coordinates:
                rows.append({
                    'die': int(die_num),
                    'block': int(block_num),
                    'x': coordinate[0],
                    'y': coordinate[1]
                })
    return pd.DataFrame(rows)

def check_blocks_relative_positions(df):
    """
    Checks if all blocks have the same relative positions of pads.

    Args:
        df (pd.DataFrame): DataFrame containing die, block, x, and y coordinates.

    Returns:
        bool: True if all blocks have the same relative positions, False otherwise.

    Example:
        is_consistent = check_blocks_relative_positions(df)
    """
    # Create a dictionary to store the relative positions of each block
    relative_positions = {}
    
    # Iterate over each block to calculate relative positions
    for block, group in df.groupby('block'):
        # Calculate relative positions within the block
        min_x, min_y = group['x'].min(), group['y'].min()
        rel_positions = group.apply(lambda row: (row['x'] - min_x, row['y'] - min_y), axis=1).tolist()
        
        # Store the relative positions as a tuple in the dictionary
        relative_positions[block] = set(rel_positions)
    
    # Check if all blocks have the same relative positions
    unique_positions = list(relative_positions.values())
    return all(pos == unique_positions[0] for pos in unique_positions)

def reassign(df, level='block', primary='y', x_asc=True, y_asc=False, inplace = False):
    """
    Reassigns level numbers (block or die) based on spatial sorting.

    Args:
        df (pd.DataFrame): DataFrame with level (block or die) and x, y coordinates.
        level (str, optional): Level to reassign ('block' or 'die'). Defaults to 'block'.
        primary (str, optional): Primary sorting axis ('x' or 'y'). Defaults to 'y'.
        x_asc (bool, optional): Sort x in ascending order. Defaults to True.
        y_asc (bool, optional): Sort y in ascending order. Defaults to False.
        inplace (bool, optional): Modify the original DataFrame. Defaults to False.

    Returns:
        pd.DataFrame or None: Modified DataFrame if inplace=False, otherwise None.

    Example:
        df = reassign(df, level='block', primary='x', x_asc=True, y_asc=True)
    """
    # Create a copy of the original DataFrame to avoid modifying it if inplace is False
    if not inplace:
        df = df.copy()

    # Extract the top-left coordinates for each level (x minimum, y maximum)
    top_left_coords = df.groupby(level).agg({'x': 'min', 'y': 'max'}).reset_index()
    
    # Determine the sorting order based on the primary axis
    if primary == 'y':
        sorted_coords = top_left_coords.sort_values(by=['y', 'x'], ascending=[y_asc, x_asc])
    else:
        sorted_coords = top_left_coords.sort_values(by=['x', 'y'], ascending=[x_asc, y_asc])
    
    # Create a mapping from old level numbers to new level numbers based on the sorted order
    sorted_coords[f'new_{level}'] = range(1, len(sorted_coords) + 1)
    level_mapping = dict(zip(sorted_coords[level], sorted_coords[f'new_{level}']))
    
    # Apply the new level numbers to the original DataFrame
    df[level] = df[level].map(level_mapping)
    
    return df if not inplace else None

def index_blocks_in_die(df, primary='y', x_asc=True, y_asc=False, inplace=False):
    """
    Assigns a unique index to blocks within each die based on spatial sorting.

    Args:
        df (pd.DataFrame): DataFrame containing die, block, x, and y coordinates.
        primary (str, optional): Primary sorting axis ('x' or 'y'). Defaults to 'y'.
        x_asc (bool, optional): Sort x in ascending order. Defaults to True.
        y_asc (bool, optional): Sort y in ascending order. Defaults to False.
        inplace (bool, optional): Modify the original DataFrame. Defaults to False.

    Returns:
        pd.DataFrame or None: Modified DataFrame if inplace=False, otherwise None.

    Example:
        df = index_blocks_in_die(df, primary='y', x_asc=True, y_asc=False)
    """
    if not inplace:
        df = df.copy()

    # Get unique dies
    die_groups = df['die'].unique()

    # Iterate over each die to classify blocks and assign numbers
    for die in die_groups:
        die_df = df[df['die'] == die]
        # Get unique blocks in sorted order (y descending, x ascending)
        
        if primary == 'y':
            unique_blocks = die_df.groupby('block').agg({'x': 'min', 'y': 'max'}).sort_values(by=['y', 'x'], ascending=[y_asc, x_asc]).reset_index()
        else:
            unique_blocks = die_df.groupby('block').agg({'x': 'min', 'y': 'max'}).sort_values(by=['x', 'y'], ascending=[x_asc, y_asc]).reset_index()
        # unique_blocks = die_df.groupby('block').agg({'x': 'min', 'y': 'max'}).sort_values(by=['y', 'x'], ascending=[False, True]).reset_index()

        # # Determine the sorting order based on the primary axis
        # if primary == 'y':
        #     sorted_coords = unique_blocks.sort_values(by=['y', 'x'], ascending=[y_asc, x_asc])
        # else:
        #     sorted_coords = unique_blocks.sort_values(by=['x', 'y'], ascending=[x_asc, y_asc])
        # # Create mapping of block to block number
        block_mapping = dict(zip(unique_blocks['block'], unique_blocks.index + 1))
        # Apply mapping to the original DataFrame
        df.loc[df['die'] == die, 'block_idx_in_die'] = df[df['die'] == die]['block'].map(block_mapping)

    # Ensure block_number is of integer type
    df['block_idx_in_die'] = df['block_idx_in_die'].astype(int)

    return df if not inplace else None

def classify_blocks_into_rows_and_columns(df, inplace = False):
    """
    Classifies blocks into rows and columns based on their coordinates.

    Args:
        df (pd.DataFrame): DataFrame containing block, x, and y coordinates.
        inplace (bool, optional): Modify the original DataFrame. Defaults to False.

    Returns:
        pd.DataFrame or None: Modified DataFrame if inplace=False, otherwise None.

    Example:
        df = classify_blocks_into_rows_and_columns(df)
    """
    # Create a copy of the original DataFrame to avoid modifying it if inplace is False
    if not inplace:
        df = df.copy()
    # Extract the top-left coordinates for each block (x minimum, y maximum)
    top_left_coords = df.groupby('block').agg({'x': 'min', 'y': 'max'}).reset_index()
    
    # Sort by y descending and x ascending to determine rows and columns
    sorted_coords = top_left_coords.sort_values(by=['y', 'x'], ascending=[False, True]).reset_index(drop=True)
    
    # Classify y-coordinates into rows
    sorted_coords['row'] = sorted_coords['y'].rank(method='dense', ascending=False).astype(int)
    
    # Classify x-coordinates into columns within each row
    sorted_coords['column'] = sorted_coords['x'].rank(method='dense').astype(int)
    
    # Create mappings for rows and columns
    row_mapping = dict(zip(sorted_coords['block'], sorted_coords['row']))
    column_mapping = dict(zip(sorted_coords['block'], sorted_coords['column']))
    
    # Apply the row and column numbers to the original DataFrame
    df['block_row'] = df['block'].map(row_mapping)
    df['block_column'] = df['block'].map(column_mapping)
    return df if not inplace else None

def classify_rows_and_columns_in_blocks(df, inplace=False):
    """
    Classifies rows and columns for pads within each block based on coordinates.

    Args:
        df (pd.DataFrame): DataFrame containing 'block', 'x', and 'y' columns.
        inplace (bool, optional): Whether to modify the original DataFrame. Defaults to False.

    Returns:
        tuple: Updated DataFrame, flag indicating consistency of row/column counts across blocks, 
               maximum row count, and maximum column count.

    Example:
        df, is_consistent, row_count, col_count = classify_rows_and_columns_in_blocks(df)
    """
    # Create a dictionary to store the row and column count of each block
    if not inplace:
        df = df.copy()
    
    # Create lists to store row and column counts for validation
    all_row_counts = []
    all_column_counts = []

    # Iterate over each block to calculate row and column count based on coordinates
    for block, group in df.groupby('block'):
        # Sort by y descending and x ascending to determine rows and columns
        sorted_coords = group.sort_values(by=['y', 'x'], ascending=[False, True]).reset_index(drop=True)
        
        # Classify y-coordinates into rows
        sorted_coords['in_block_row'] = sorted_coords['y'].rank(method='dense', ascending=False).astype(int)
        
        # Classify x-coordinates into columns within each row
        sorted_coords['in_block_column'] = sorted_coords['x'].rank(method='dense').astype(int)
        
        # Create (x, y) tuples as dictionary keys for mapping
        row_mapping = dict(zip(zip(sorted_coords['x'], sorted_coords['y']), sorted_coords['in_block_row'].astype(int)))
        column_mapping = dict(zip(zip(sorted_coords['x'], sorted_coords['y']), sorted_coords['in_block_column'].astype(int)))

        # Get the indices of the current block in the original DataFrame
        block_indices = df[df['block'] == block].index

        # Extract (x, y) tuples for the current block
        xy_tuples = list(zip(df.loc[block_indices, 'x'], df.loc[block_indices, 'y']))

        # Use the mapping dictionaries to update the rows and columns for the current block
        df.loc[block_indices, 'in_block_row'] = [int(row_mapping[xy]) for xy in xy_tuples]
        df.loc[block_indices, 'in_block_column'] = [int(column_mapping[xy]) for xy in xy_tuples]

        # Append row and column counts for validation
        all_row_counts.append(sorted_coords['in_block_row'].nunique())
        all_column_counts.append(sorted_coords['in_block_column'].nunique())
    row_count = max(all_row_counts)
    column_count = max(all_column_counts)

    same_row_col_counts = len(set(all_row_counts)) == 1 and len(set(all_column_counts)) == 1
    # # Check if all blocks have the same row and column counts
    # # if len(set(all_row_counts)) == 1 and len(set(all_column_counts)) == 1:
    # if same_row_col_counts:
    #     print("All blocks have the same row and column counts.")
    # else:
    #     print("Warning: Not all blocks have the same row and column counts.")
    #     print(f"Row counts: {all_row_counts}")
    #     print(f"Column counts: {all_column_counts}")

    # Ensure the final columns are converted to int type
    df['in_block_row'] = df['in_block_row'].astype(int)
    df['in_block_column'] = df['in_block_column'].astype(int)

    return df, same_row_col_counts, row_count, column_count

def apply_labels_to_blocks(df, row_labels, row_label_meaning, column_labels, column_label_meaning, inplace=False):
    """
    Applies custom labels to rows and columns in blocks.

    Args:
        df (pd.DataFrame): DataFrame containing 'in_block_row' and 'in_block_column'.
        row_labels (list): List of labels for rows.
        row_label_meaning (str): Name for the new row label column.
        column_labels (list): List of labels for columns.
        column_label_meaning (str): Name for the new column label column.
        inplace (bool, optional): Whether to modify the original DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: Updated DataFrame with new row and column label columns.

    Example:
        df = apply_labels_to_blocks(df, ['A', 'B'], 'row_label', [1, 2], 'col_label')
    """
    if not inplace:
        df = df.copy()
    # Check if all blocks have consistent row and column counts
    row_count = df['in_block_row'].nunique()
    column_count  = df['in_block_column'].nunique()
    # Validate that the provided labels match the row and column counts
    if len(row_labels) != row_count:
        raise ValueError(f"Number of row labels ({len(row_labels)}) does not match row count ({row_count})")
    if len(column_labels) != column_count:
        raise ValueError(f"Number of column labels ({len(column_labels)}) does not match column count ({column_count})")
    
    # Create mappings for row and column labels
    row_mapping = dict(zip(range(1, row_count + 1), row_labels))
    column_mapping = dict(zip(range(1, column_count + 1), column_labels))
    
    # Apply the row and column labels to the DataFrame
    df[row_label_meaning] = df['in_block_row'].map(row_mapping)
    df[column_label_meaning] = df['in_block_column'].map(column_mapping)
    
    return df

def process_for_plotting(df):
    """
    Prepares block and die bounding box data for plotting.

    Args:
        df (pd.DataFrame): DataFrame containing 'block', 'die', 'x', and 'y' columns.

    Returns:
        tuple: DataFrames containing bounding box information for dies and blocks.

    Example:
        dies_info, blocks_info = process_for_plotting(df)
    """
    # Calculate the min and max coordinates for each block to determine the bounding box
    block_groups = df.groupby('block')
    blocks_info = block_groups.agg({'x': ['min', 'max'], 'y': ['min', 'max']})
    blocks_info.columns = ['x_min', 'x_max', 'y_min', 'y_max']
    blocks_info.reset_index(inplace=True)

    # Calculate the min and max coordinates for each die to determine the bounding box
    die_groups = df.groupby('die')
    dies_info = die_groups.agg({'x': ['min', 'max'], 'y': ['min', 'max']})
    dies_info.columns = ['x_min', 'x_max', 'y_min', 'y_max']
    dies_info.reset_index(inplace=True)

    return dies_info, blocks_info

# Function to load a JSON file
def save_to_json(filename, content):
    with open(filename, 'w') as json_file:
        json.dump(content, json_file, indent=4)


# Function to replace engineering symbols with scientific notation
def replace_engineering_symbols(element):
    """
    Replaces engineering symbols (e.g., 'k', 'M') in a string with scientific notation.

    Args:
        element (str): Input string with engineering symbols.

    Returns:
        str: String with engineering symbols replaced by scientific notation.

    Example:
        result = replace_engineering_symbols("1k")  # result = "1e3"
    """
    element = element.strip()
    element = re.sub(r'([0-9.]+)k', r'\1e3', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)m', r'\1e-3', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)u', r'\1e-6', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)M', r'\1e6', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)G', r'\1e9', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)g', r'\1e9', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)n', r'\1e-9', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)p', r'\1e-12', element, flags=re.IGNORECASE)
    element = re.sub(r'([0-9.]+)T', r'\1e12', element, flags=re.IGNORECASE)
    return element

# Function to convert a string input into a list of numbers
def convert_string_to_list(input_string):
    """
    Converts a comma-separated string into a list of floats, supporting engineering notation.

    Args:
        input_string (str): Input string with comma-separated values.

    Returns:
        list: List of converted float values.

    Example:
        result = convert_string_to_list("1k, 2M, 3")
    """
    # Split the string by commas to get individual elements
    string_elements = input_string.split(',')
    number_list = []
    
    for element in string_elements:
        if element.strip() == '':
            continue # Skip empty elements
        try:
            # Replace engineering symbols and convert to float
            number = float(replace_engineering_symbols(element))
            number_list.append(number)
        except ValueError:
            # print(f"Warning: '{element.strip()}' is not a valid number and will be ignored.")
            st.warning(f"Warning: '{element.strip()}' is not in valid format and will be ignored.") # to specifically show the warning in the streamlit app
    return number_list

def check_pin_cfgs(layout_data):
    """
    Validates the pin configurations in the layout data.

    Args:
        layout_data (dict): A dictionary containing a 'pin_cfgs' key with a list of pin names.

    Behavior:
        - Checks if all pin names end with '_P' or '_N'.
        - Verifies that each pin with '_P' has a corresponding '_N' with the same prefix.

    Warnings:
        - Issues warnings if a pin name does not end with '_P' or '_N'.
        - Warns if there are unmatched '_P' or '_N' pins.

    Example:
        layout_data = {'pin_cfgs': ['A_P', 'A_N', 'B_P']}
        check_pin_cfgs(layout_data)
    """
    pin_cfgs = layout_data.get('pin_cfgs', [])

    # check if all items in the list end with '_P' or '_N'
    for item in pin_cfgs:
        if not (item.endswith('_P') or item.endswith('_N')):
            # print(f"Warning: '{item}' does not end with '_P' or '_N'. Corrsponding measurements result may be affected.")
            st.warning(f"Warning: '{item}' does not end with '_P' or '_N'. Corrsponding measurements result may be affected.")
            exit()

    # create two sets to store the prefixes of the pin configurations
    prefix_p = set()
    prefix_n = set()

    # iterate through the pin configurations
    for item in pin_cfgs:
        prefix = item[:-2]
        if item.endswith('_P'):
            prefix_p.add(prefix)
        elif item.endswith('_N'):
            prefix_n.add(prefix)

    # check if all items ending with '_P' have a corresponding item ending with '_N'
    if prefix_p != prefix_n:
        # print("Warning: There are pin configurations ending with '_P' that do not have a corresponding '_N', or vice versa. Corrsponding measurements result may be affected.")
        st.warning("Warning: There are pin configurations ending with '_P' that do not have a corresponding '_N', or vice versa. Corrsponding measurements result may be affected.")

def dropout_columns(df, columns, inplace=False):
    """
    Drops specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to drop.
        inplace (bool, optional): Whether to modify the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped, unless inplace=True.

    Example:
        df = dropout_columns(df, ['column1', 'column2'])
    """
    if not inplace:
        df = df.copy()
    for col in columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def divide_data_by_columns(data, columns):
    """
    Divides a list of data into sublists based on the specified number of columns.

    Args:
        data (list): The list of data to be divided.
        columns (int): The number of columns to divide the data into.

    Returns:
        list of lists: A list containing sublists of data.

    Example:
        divided_data = divide_data_by_columns([1, 2, 3, 4, 5, 6], 3)
        # Result: [[1, 4], [2, 5], [3, 6]]
    """
    return [data[i::columns] for i in range(columns)]

def df_filter(df, filter_dict, inplace=False):
    """
    Filters rows in a DataFrame based on a dictionary of conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filter_dict (dict): A dictionary where keys are column names and values are filter conditions.
                            - If the value is a list or set, rows are filtered using `isin`.
                            - Otherwise, rows are filtered using equality.
        inplace (bool, optional): Whether to modify the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame: The filtered DataFrame, unless inplace=True.

    Example:
        filtered_df = df_filter(df, {'column1': [1, 2], 'column2': 'A'})
    """
    # # if the certrain key is not in the filter_dict, then the filter is not applied
    # for key in filter_dict:
    #     if key in df.columns:
    #         df = df[df[key] == filter_dict[key]]
    # return df
    if not inplace:
        df = df.copy()
    for key, values in filter_dict.items():
        if key in df.columns:
            # if the values are a list or set, then use the isin filter
            if isinstance(values, (list, set)):
                df = df[df[key].isin(values)]
            else:
                # otherwise, use the equality filter
                df = df[df[key] == values]
    return df