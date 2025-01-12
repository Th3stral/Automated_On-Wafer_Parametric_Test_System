import streamlit as st
import numpy as np
import pandas as pd
import json
import re

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import subprocess

import pickle
import copy

import config as sys_config

from utils.general_utils import save_to_json, df_filter, convert_string_to_list, check_pin_cfgs, dropout_columns, divide_data_by_columns


# set up the general configurations
pin_variation_too_long_msg = sys_config.PIN_VARIANTS_TOO_LONG_MSG
current_variation_too_long_msg = sys_config.CURR_VARIANTS_TOO_LONG_MSG

pin_map_file = sys_config.PIN_MAP_FILE_PATH
curr_cfg_file = sys_config.CURR_CFG_FILE_PATH
cfg_maps_file = sys_config.CFG_MAPS_FILE_PATH
structure_folder_path = sys_config.WAFER_STRUCTURE_FOLDER_PATH

# read existing designs
file_names = os.listdir(structure_folder_path)

csv_files = [f for f in file_names if os.path.isfile(os.path.join(structure_folder_path, f)) and f.endswith('.csv')]
design_names = [f.replace('_wafer_structure.csv', '') for f in csv_files]
# design = st.selectbox('Select Wafer Design', design_names)

st.title("Create New Wafer Mapping Event")

# define the double check window
@st.dialog("Do you wish to continue?")
def double_check_window(message):
    st.warning(message)
    st.write("### Please confirm your choice.")
    
    if st.button("Yes"):
        st.rerun()
    elif st.button("No", type='primary'):
        st.success("You may now close this window and make the necessary changes.")
        st.stop()
    st.stop()


## load measurement config files
if "pin_cfgs" not in st.session_state:
    try:
        with open(pin_map_file, 'r') as file:
            pin_map_file = json.load(file)
            st.session_state.pin_cfgs = pin_map_file
    except Exception as e:
        st.warning("No pin configurations file found, initialize an empty one")
        st.session_state.pin_cfgs = {}

if "all_curr_cfgs" not in st.session_state:
    try:
        with open(curr_cfg_file, 'r') as file:
            exist_curr_cfgs = json.load(file)
            st.session_state.all_curr_cfgs = exist_curr_cfgs
    except Exception as e:
        st.warning("No current configurations file found, initialize an empty one")
        st.session_state.all_curr_cfgs = {}

if "all_cfg_maps" not in st.session_state:
    try:
        with open(cfg_maps_file, 'r') as file:
            exist_cfg_maps = json.load(file)
            st.session_state.all_cfg_maps = exist_cfg_maps
    except Exception as e:
        st.warning("No configuration maps file found, initialize an empty one")
        st.session_state.all_cfg_maps = {}

# init the session state variables
if 'map_to_issue' not in st.session_state:
    st.session_state.map_to_issue = {}

if "filter_dict" not in st.session_state:
    st.session_state.filter_dict = {}


if 'cache_design' not in st.session_state:
    st.session_state.cache_design = design_names[0]

if 'cfg_maps' not in st.session_state:
    st.session_state.cfg_maps = {}

# select the wafer design
def save_design():
    st.session_state.cache_design = st.session_state.k_wafer_design

wafer_design = st.selectbox('Select Wafer Design', design_names, index=design_names.index(st.session_state.cache_design) if st.session_state.cache_design in design_names else 0
                            ,key='k_wafer_design', on_change=save_design)
df = pd.read_csv(os.path.join(structure_folder_path, f"{wafer_design}_wafer_structure.csv"))

if "current_design" not in st.session_state:
    st.session_state.current_design = wafer_design

# wafer design changed
if wafer_design != st.session_state.current_design:
    print("Design changed")
    # read the new wafer design
    st.session_state.current_design = wafer_design
    df = pd.read_csv(os.path.join(structure_folder_path, f"{wafer_design}_wafer_structure.csv"))
    
    # update the saved configuration maps
    if st.session_state.current_design not in st.session_state.all_curr_cfgs:
        st.session_state.all_curr_cfgs[st.session_state.current_design] = {}
    if st.session_state.curr_variation not in st.session_state.all_curr_cfgs[st.session_state.current_design]:
        st.session_state.all_curr_cfgs[st.session_state.current_design][st.session_state.curr_variation] = {}
    st.session_state.design_curr_cfgs = st.session_state.all_curr_cfgs[st.session_state.current_design][st.session_state.curr_variation]

    # deleate the assigned cfg_maps
    del st.session_state.cfg_maps
    st.session_state.filter_dict.clear() # clear the filter dictionary
    st.session_state.map_to_issue.clear() # clear the map to issue dictionary


    # update and initialize the configuration maps
    if st.session_state.current_design not in st.session_state.all_cfg_maps:
        st.session_state.all_cfg_maps[st.session_state.current_design] = {}
    if st.session_state.pin_variation not in st.session_state.all_cfg_maps[st.session_state.current_design]:
        st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation] = {}
    if st.session_state.curr_variation not in st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation]:
        st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation] = {}

    st.rerun()

# the variation dataframe for variation selection
variation_df = df.copy()



# variation_df = dropout_columns(variation_df, ['wafer', 'design', 'x', 'y', 'block', 'die', 'block_row', 'block_column'], inplace=False)
variation_df = dropout_columns(variation_df, sys_config.VARIATION_OBJECT_IGNO_LIST, inplace=False)


##### new code that remembes the previous selected variation objects
if 'cache_curr_variation' not in st.session_state:
    st.session_state.cache_curr_variation = variation_df.columns[0]

if 'cache_pin_variation' not in st.session_state:
    st.session_state.cache_pin_variation = variation_df.columns[1] if len(variation_df.columns) > 1 else variation_df.columns[0]

def curr_variation_changed():

    st.session_state.cache_curr_variation = st.session_state.k_curr_variation
    if st.session_state.cache_pin_variation == st.session_state.cache_curr_variation:
        # if the previous pin_variation conflicts with the new curr_variation, reselect an available option
        remaining_options = variation_df.columns.drop(st.session_state.cache_curr_variation)
        st.session_state.cache_pin_variation = remaining_options[0] if not remaining_options.empty else ""

def pin_variation_changed():
    st.session_state.cache_pin_variation = st.session_state.k_pin_variation

# select current variation object and pin variation object: 
curr_variation_options = variation_df.columns
curr_variation = st.selectbox(
    label='Select current variation object',
    options=curr_variation_options,
    index=curr_variation_options.get_loc(st.session_state.cache_curr_variation) if st.session_state.cache_curr_variation in curr_variation_options else 0,
    key='k_curr_variation',
    on_change=curr_variation_changed,
    help="Select the variation that does not requier different pin configurations but requiers different force current configurations, e.g. material, thickness, etc."
)

pin_variation_options = variation_df.columns.drop(st.session_state.k_curr_variation)
pin_variation = st.selectbox(
    label='Select pin variation object',
    options=pin_variation_options,
    index=pin_variation_options.get_loc(st.session_state.cache_pin_variation) if st.session_state.cache_pin_variation in pin_variation_options else 0,
    key='k_pin_variation',
    on_change=pin_variation_changed,
    help="Select the variation that requiers different pin configurations, e.g. structure layout, etc."
)
### new code ends here

##### old code if needed
# curr_variation = st.selectbox('Select current variation object', variation_df.columns, index=0, help="Select the variation that does not requier different pin configurations but requiers different force current configurations, e.g. material, thickness, etc.")
# pin_variation = st.selectbox('Select pin variation object', variation_df.columns.drop(curr_variation), index=0, help="Select the variation that requiers different pin configurations, e.g. structure layout, etc.")
##### old code ends here

# list variants
unique_curr_variation = variation_df[str(curr_variation)].unique()
unique_curr_variation = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in unique_curr_variation]


unique_pin_variation = variation_df[str(pin_variation)].unique()
unique_pin_variation = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in unique_pin_variation]

# check if the number of unique variations is too long
if "curr_variation" not in st.session_state:

    st.session_state.curr_variation = curr_variation
    if variation_df[str(curr_variation)].nunique() >= 10:
        # if the number of unique variations is too long, then show a double check window
        double_check_window(current_variation_too_long_msg)
    
if "new_current_cfg" not in st.session_state:
    st.session_state.new_current_cfg = {}

if "pin_variation" not in st.session_state:
    st.session_state.pin_variation = pin_variation

    if variation_df[str(pin_variation)].nunique() >= 10:
        # if the number of unique variations is too long, then show a double check window
        double_check_window(pin_variation_too_long_msg)


# design_curr_cfgs is a dictionary that stores all the current configurations for the current selected design
if "design_curr_cfgs" not in st.session_state:
    # initialize the dictionary
    st.session_state.current_design = wafer_design
    if st.session_state.current_design not in st.session_state.all_curr_cfgs: # check if the design is in the current configurations
        st.session_state.all_curr_cfgs[st.session_state.current_design] = {} # if not, add it
    if st.session_state.curr_variation not in st.session_state.all_curr_cfgs[st.session_state.current_design]: # check if the variation is in the current configurations
        st.session_state.all_curr_cfgs[st.session_state.current_design][st.session_state.curr_variation] = {} # if not, add it
    # get the curr cfgs corresponding to the selected design
    st.session_state.design_curr_cfgs = st.session_state.all_curr_cfgs[st.session_state.current_design][st.session_state.curr_variation]

# if the current variation is changed, then update the current variation
if curr_variation != st.session_state.curr_variation:
    # print("Variation changed")
    st.session_state.curr_variation = curr_variation # update the current variation

    # update and initialize the configuration maps for every design, pin variation and current variation
    if st.session_state.current_design not in st.session_state.all_cfg_maps:
        st.session_state.all_cfg_maps[st.session_state.current_design] = {}
    if st.session_state.pin_variation not in st.session_state.all_cfg_maps[st.session_state.current_design]:
        st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation] = {}
    if st.session_state.curr_variation not in st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation]:
        st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation] = {}

    # update the saved configuration maps
    if st.session_state.curr_variation not in st.session_state.all_curr_cfgs[st.session_state.current_design]: # check if the variation is in the current configurations
        st.session_state.all_curr_cfgs[st.session_state.current_design][st.session_state.curr_variation] = {} # if not, add it
    st.session_state.design_curr_cfgs = st.session_state.all_curr_cfgs[st.session_state.current_design][st.session_state.curr_variation] # get the curr cfgs corresponding to the selected design

    # clear necessary session state variables
    st.session_state.new_current_cfg.clear() # clear the new current configuration
    st.session_state.map_to_issue.clear()
    st.session_state.filter_dict.clear()
    # check if the number of unique variations is too long
    if variation_df[str(curr_variation)].nunique() >= 10:
        double_check_window(current_variation_too_long_msg)

    
    

if pin_variation != st.session_state.pin_variation:
    print("Pin Variation changed")
    st.session_state.pin_variation = pin_variation

    # update and initialize the configuration maps for every design, pin variation and current variation    
    if st.session_state.current_design not in st.session_state.all_cfg_maps:
        st.session_state.all_cfg_maps[st.session_state.current_design] = {}
    if st.session_state.pin_variation not in st.session_state.all_cfg_maps[st.session_state.current_design]:
        st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation] = {}
    if st.session_state.curr_variation not in st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation]:
        st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation] = {}

    # clear necessary session state variables
    del st.session_state.cfg_maps # clear the assigned cfg_maps
    st.session_state.map_to_issue.clear()   
    st.session_state.filter_dict.clear()

    # check if the number of unique variations is too long
    if variation_df[str(pin_variation)].nunique() >= 10:
        double_check_window(pin_variation_too_long_msg)

##### current cfg management section #####
# fragement for creating new current configurations for the selected design, current variation
@st.fragment
def material_current_cfg():
    with st.expander(f"Create New Force Current Configuration for {st.session_state.curr_variation}"):

        ext_border = st.container(border=True)
        # cols for the input and view
        settings, view = ext_border.columns([2, 1], vertical_alignment="bottom")
        with settings:
            # set to two columns
            col_num = 2
            columns_data = divide_data_by_columns(unique_curr_variation, col_num)
            st.write(f"## Force Current Config for {st.session_state.curr_variation}")
            # display logic for current input
            columns = st.columns(col_num)
            # iterate through the 2 columns
            for col_index, col_data in enumerate(columns_data):
                with columns[col_index]:
                    # for individual input block
                    for i, item in enumerate(col_data):
                        # convert the item to int if it is a numpy int for json serialization
                        if isinstance(item, np.int64):
                            item = int(item)
                        # value input
                        new_current = st.text_input(f'Enter force current values for {st.session_state.curr_variation} {item} in Amp')
                        # convert the string to a list
                        st.session_state.new_current_cfg[item] = convert_string_to_list(new_current)
                        # check if the values are greater than 1, give warning for high current values
                        if any(value >= 1 for value in st.session_state.new_current_cfg[item]):
                            st.warning("High current values detected, please double check make sure the values are in the correct format.")
                        st.write(st.session_state.new_current_cfg[item])
            # enter the name for the current configuration
            name = st.text_input("Configuration Name")
            # save
            if st.button('Save Config'):
                # print(st.session_state.current_config)
                if name is not None and name.strip() != "":
                    # save current configuration
                    st.session_state.design_curr_cfgs[name] = dict(st.session_state.new_current_cfg) # dict() is used to create a copy of the dictionary
                    # refresh the current configurations
                    st.session_state.all_curr_cfgs[wafer_design][st.session_state.curr_variation] = st.session_state.design_curr_cfgs
                    # st.session_state.all_curr_cfgs[wafer_design] = st.session_state.design_curr_cfgs
                    # save to json file
                    try:
                        save_to_json(curr_cfg_file, st.session_state.all_curr_cfgs)
                    except Exception as e:
                        st.exception(e)
                        st.stop()
                    st.success("Configuration saved successfully")
                    st.rerun()
                else:
                    st.warning("Please enter a name for this current configuration")
        # for viewing the existing current configurations
        with view:
            st.write("## Existing Configs for selected design")
            crr_cfg_df = pd.DataFrame.from_dict(st.session_state.design_curr_cfgs, orient='index').map(lambda x: x if isinstance(x, list) else [x])
            st.dataframe(crr_cfg_df)
            # select a configuration to delete
            to_del = st.selectbox("Select a configuration to detele", list(st.session_state.design_curr_cfgs.keys()), index=None, placeholder="Select a configuration to delete")
            if st.button("Delete"):
                if to_del in st.session_state.design_curr_cfgs:
                    del st.session_state.design_curr_cfgs[to_del]
                    # st.session_state.all_curr_cfgs[wafer_design] = st.session_state.design_curr_cfgs
                    save_to_json(curr_cfg_file, st.session_state.all_curr_cfgs)
                    st.rerun()
                elif to_del is None:
                    st.warning("Please select a configuration to delete")
                    st.stop()        
# call fragment
material_current_cfg()
    
##### Configuration assignment section #####
# fragment for assigning configurations to structures
@st.fragment
def assign_to_structures():
    with st.expander(f"Assign Configurations to {st.session_state.pin_variation}"):
        pin_var_container = st.container()
        pin_var_cols = pin_var_container.columns([7, 2])
        with pin_var_cols[0]:

            ##########old code###############
            # columns = st.columns(len(unique_pin_variation))
            # for i, structure in enumerate(unique_pin_variation):
            #######################################
            ##########new code####################
            # new ver limits the number of columns to 3 to avoui the layout being messed up
            col_num = 3
            columns_data = divide_data_by_columns(unique_pin_variation, col_num)

            st.write("## Selectboxes")
            columns = st.columns(col_num)
            for i, col_data in enumerate(columns_data):
                with columns[i].container():
                    for i, structure in enumerate(col_data):
            #######################################
                        if isinstance(structure, (np.int64, np.int32)):
                            structure = int(structure)
                        st.write(f'## {pin_variation} {structure}')
                        
                        # Check if the structure already exists in the state, if not initialize it
                        if 'cfg_maps' not in st.session_state:
                            st.session_state.cfg_maps = {}

                        if structure not in st.session_state.cfg_maps:
                            st.session_state.cfg_maps[structure] = {}

                        if st.toggle(f'{pin_variation} {structure} is integrated ?', value=False):
                            st.write('Integrated')
                            number = st.number_input(f'Number of layouts for {pin_variation} {structure}', min_value=1, max_value=10, value=2, step=1)

                            # Update the structure configurations
                            current_cfgs = st.session_state.cfg_maps[structure]

                            # Clear out any entries beyond the current number
                            keys_to_remove = [k for k in current_cfgs if k >= number]
                            for key in keys_to_remove:
                                del st.session_state.cfg_maps[structure][key]

                            # Create or update the necessary entries
                            for k in range(number):
                                st.write(f'Layout {k+1}')
                                if k not in st.session_state.cfg_maps[structure]:
                                    st.session_state.cfg_maps[structure][k] = {}

                                # Set alias name with a default value as layout1, layout2, etc.
                                default_alias = f'layout{k+1}'
                                st.session_state.cfg_maps[structure][k]['alias'] = st.text_input(
                                    f'Alias name for layout {k+1} ({pin_variation} {structure})', 
                                    value=default_alias
                                )

                                if st.session_state.cfg_maps[structure][k]['alias'] == '':
                                    st.warning("Please enter an alias name for the layout or default name will be used")
                                    st.session_state.cfg_maps[structure][k]['alias'] = default_alias

                                # Set pin configurations
                                st.session_state.cfg_maps[structure][k]['pin_cfgs'] = st.multiselect(
                                    f'Select pin configuration {k+1} for {structure}', 
                                    list(st.session_state.pin_cfgs.keys())
                                )
                                
                                # Set current configuration
                                st.session_state.cfg_maps[structure][k]['curr_cfg'] = st.selectbox(
                                    f'Select Current configuration {k+1} for {pin_variation} {structure}', 
                                    list(st.session_state.design_curr_cfgs.keys())
                                )
                                check_pin_cfgs(st.session_state.cfg_maps[structure][k])

                                # compliance voltage for SMu
                                st.session_state.cfg_maps[structure][k]['comp_v'] = st.number_input(
                                    f'Enter compliance voltage for {pin_variation} {structure} layout {k+1}',
                                    min_value=0.0, max_value=100.0, value=1.0, step=0.1
                                )

                                # Set NPLC
                                st.session_state.cfg_maps[structure][k]['nplc'] = st.number_input(
                                    f'Enter NPLC for {pin_variation} {structure} layout {k+1}',
                                    min_value=0.0, max_value=10.0, value=5.0, step=1.0,
                                    help="Number of Power Line Cycles"
                                )

                                # Set number of readings
                                st.session_state.cfg_maps[structure][k]['nrdgs'] = st.number_input(
                                    f'Enter number of readings for {pin_variation} {structure} layout {k+1}',
                                    min_value=1, max_value=100, value=1, step=1
                                )


                        else:
                            # Clear the dictionary for the current structure if not integrated
                            st.write('Not integrated')
                            st.session_state.cfg_maps[structure] = {
                                0: {
                                    'alias': structure,
                                    'pin_cfgs': st.multiselect(
                                        f'Select pin configuration for {pin_variation} {structure}', 
                                        list(st.session_state.pin_cfgs.keys())
                                    ),
                                    'curr_cfg': st.selectbox(
                                        f'Select Current configuration for {pin_variation} {structure}', 
                                        list(st.session_state.design_curr_cfgs.keys())
                                    ),
                                    'comp_v': st.number_input(
                                        f'Enter compliance voltage for {pin_variation} {structure}',
                                        min_value=0.0, max_value=100.0, value=1.0, step=0.1
                                    ),
                                    'nplc': st.number_input(
                                        f'Enter NPLC for {pin_variation} {structure}',
                                        min_value=0.0, max_value=10.0, value=5.0, step=1.0,
                                        help="Number of Power Line Cycles"
                                    ),
                                    'nrdgs': st.number_input(
                                        f'Enter number of readings for {pin_variation} {structure}',
                                        min_value=1, max_value=100, value=1, step=1
                                    ),
                                    # 'repeat': st.number_input(
                                    #     f'Enter number of repeats for {pin_variation} {structure}',
                                    #     min_value=1, max_value=100, value=1, step=1
                                    # )
                                }
                            }
                            check_pin_cfgs(st.session_state.cfg_maps[structure][0])
        with pin_var_cols[1]:
            st.write(st.session_state.cfg_maps)
            new_cfg_map_name = st.text_input("Enter a name for the configuration", key='cfg_name')
            if new_cfg_map_name in st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation]:
                st.warning("IGNORE if the configuration has just been saved: Configuration name already exists, continue saving will overwrite the existing configuration")
            if st.button("Save"):
                if not new_cfg_map_name is None and new_cfg_map_name.strip() != "":
                    # save to json file
                    try:
                        with open(cfg_maps_file, 'r') as file:
                            exist_cfg_maps = json.load(file)
                            st.session_state.all_cfg_maps = exist_cfg_maps
                            if st.session_state.current_design not in st.session_state.all_cfg_maps:
                                st.session_state.all_cfg_maps[st.session_state.current_design] = {}
                            if st.session_state.pin_variation not in st.session_state.all_cfg_maps[st.session_state.current_design]:
                                st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation] = {}
                            if st.session_state.curr_variation not in st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation]:
                                st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation] = {}
                            
                            st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation][new_cfg_map_name] = st.session_state.cfg_maps
                            save_to_json(cfg_maps_file, st.session_state.all_cfg_maps)
                            st.success("Configurations saved successfully")
                            st.rerun()
                    except Exception as e:
                        st.exception(e)
                        st.stop()
                else:
                    st.warning("Please enter a name for the configuration")
# call fragment
assign_to_structures()

#### general event cfg section ####
# Final setup before testing
issue_testing_container = st.container(border=True)
with issue_testing_container:
    ##### load config for variations
    st.write("## Variation Configuration")
    if 'map_to_issue' not in st.session_state:
        st.session_state.map_to_issue = {}
    with st.expander("Load Configurations for Variations", expanded=True):
        cfg_map_issue_container = st.container()
        cfg_map_issue_cols = cfg_map_issue_container.columns(2)
        with cfg_map_issue_cols[0]:
            st.write("### Continue with previously assigned configurations")
            if st.button('Continue'):
                st.session_state.map_to_issue = copy.deepcopy(st.session_state.cfg_maps)
        with cfg_map_issue_cols[1]:
            st.write("### Or Load from saved configurations")
            cfg_map = st.selectbox("Select Configurations", list(st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation].keys()), index=None, placeholder="Select a configuration")
            if st.button('Load', disabled=cfg_map is None):
                st.session_state.map_to_issue = copy.deepcopy(st.session_state.all_cfg_maps[st.session_state.current_design][st.session_state.pin_variation][st.session_state.curr_variation][cfg_map])

        col_num = 3

        # to divide the data into columns
        columns_data = divide_data_by_columns(list(st.session_state.map_to_issue.items()), col_num)
        st.write("## Loaded Configurations")
        if st.session_state.map_to_issue == {}:
            st.warning("No configurations has been loaded yet")
        columns = st.columns(col_num)

        # iter through the columns
        for i, column_data in enumerate(columns_data):
            with columns[i].container():
                # for each configuration
                for structure, layouts in column_data:
                    st.write(f"### {structure}")
                    for layout, config_prof in layouts.items():
        # for structure, layouts in st.session_state.map_to_issue.items():
            # st.write(f"### {structure}")
            # for layout, config_prof in layouts.items():
                        st.write(f"#### Layout {int(layout)+1}")
                        # alias = config_prof['alias']
                        alias = config_prof.get('alias', None)
                        # curr_cfg_name = config_prof['curr_cfg']
                        curr_cfg_name = config_prof.get('curr_cfg', None)
                        # pin_cfg_names = config_prof['pin_cfgs']
                        pin_cfg_names = config_prof.get('pin_cfgs', None)
                        st.write(f"alias: {alias}")

                        nplc = config_prof.get('nplc', 5.0)
                        comp_v = config_prof.get('comp_v', 1.0)
                        nrdgs = config_prof.get('nrdgs', 1)
                        # repeat = config_prof.get('repeat', 1)

                        if curr_cfg_name is not None:
                            if curr_cfg_name in st.session_state.design_curr_cfgs:
                                curr_cfg_to_issue = st.session_state.design_curr_cfgs[curr_cfg_name]
                            else:
                                # check if the current configuration is available
                                st.error(f"Current configuration ''{curr_cfg_name}'' not found, it may have been deleted, please reassign the configuration")
                                st.stop()
                        else:
                            curr_cfg_to_issue = None
                        st.write(f"current variation object: ''{st.session_state.curr_variation}''")
                        st.write(f"current configuration name:", curr_cfg_name)
                        st.write(f"force current values in Amp:", curr_cfg_to_issue)

                        # double check if the pin configurations are available
                        for name in pin_cfg_names:
                            if name not in st.session_state.pin_cfgs:
                                st.error(f"Pin configuration ''{name}'' not found, it may have been deleted, please reassign the configuration")
                                st.stop()
                        st.write(f"pin variation object: ''{st.session_state.pin_variation}''")
                        st.write(f"pin configurations names:")
                        st.write(pin_cfg_names)
                        st.write(f"compliance voltage in V:", comp_v)
                        st.write(f"NPLC:", nplc)
                        st.write(f"number of readings:", nrdgs)
                        # st.write(f"number of repeats:", repeat)

    st.write("## Event Configuration")
    filter_col, home_col = st.columns([3, 1])
    with filter_col:
        with st.expander("Filter settings", expanded=True):
            # Filter the data based on the selected criteria

            if 'filter_dict' not in st.session_state:
                st.session_state.filter_dict = {}

            filter_dict = {}
            if 'cache_filter_die_togg' not in st.session_state:
                st.session_state.cache_filter_die_togg = True
            def die_filter_changed():
                st.session_state.cache_filter_die_togg = st.session_state.k_filter_die_togg
            
            if 'cache_filter_die_choice' not in st.session_state:
                st.session_state.cache_filter_die_choice = []
            def die_filter_choice_changed():
                st.session_state.cache_filter_die_choice = st.session_state.k_filter_die_choice

            if df['die'].nunique() > 1:
                # if not st.toggle("Test all dies", value=True):
                    # dies = st.multiselect('Select Dies to test', df['die'].unique())    
                if not st.toggle("Test all dies", value=st.session_state.cache_filter_die_togg, key='k_filter_die_togg', on_change=die_filter_changed):
                    dies = st.multiselect('Select Dies to test', df['die'].unique())    
                    # dies = st.multiselect(
                    #     label='Select Dies to test', 
                    #     options=df['die'].unique(), 
                    #     default=st.session_state.cache_filter_die_choice if all(item in df['die'].unique() for item in st.session_state.cache_filter_die_choice) else [],
                    #     key='k_filter_die_choice', 
                    #     on_change=die_filter_choice_changed
                    # )      
                    filter_dict['die'] = dies
                else:
                    if 'die' in filter_dict:
                        del filter_dict['die']
            else:
                if 'die' in filter_dict:
                    del filter_dict['die']

            if 'cache_filter_block_togg' not in st.session_state:
                st.session_state.cache_filter_block_togg = True
            def block_filter_changed():
                st.session_state.cache_filter_block_togg = st.session_state.k_filter_block_togg
            
            if 'cache_filter_block_choice' not in st.session_state:
                st.session_state.cache_filter_block_choice = []
            def block_filter_choice_changed():
                st.session_state.cache_filter_block_choice = st.session_state.k_filter_block_choice

            # if not st.toggle(f"Test all in die blocks", value=True):
                # blocks = st.multiselect('Select Blocks to test', df['block_idx_in_die'].unique())
            if not st.toggle("Test all in die blocks", value=st.session_state.cache_filter_block_togg, key='k_filter_block_togg', on_change=block_filter_changed):
                blocks = st.multiselect('Select Blocks to test', df['block_idx_in_die'].unique())
                # blocks = st.multiselect(
                #     label='Select Blocks to test', 
                #     options=df['block_idx_in_die'].unique(), 
                #     default=st.session_state.cache_filter_block_choice if all(item in df['block_idx_in_die'].unique() for item in st.session_state.cache_filter_block_choice) else [],
                #     key='k_filter_block_choice', 
                #     on_change=block_filter_choice_changed
                # )
                filter_dict['block_idx_in_die'] = blocks
            else:
                if 'block_idx_in_die' in filter_dict:
                    del filter_dict['block_idx_in_die']

            
            if 'cache_filter_currvar_togg' not in st.session_state:
                st.session_state.cache_filter_currvar_togg = True
            def curr_variation_filter_changed():
                st.session_state.cache_filter_currvar_togg = st.session_state.k_filter_currvar_togg
            
            if 'cache_filter_currvar_choice' not in st.session_state:
                st.session_state.cache_filter_currvar_choice = []
            def curr_variation_filter_choice_changed():
                st.session_state.cache_filter_currvar_choice = st.session_state.k_filter_currvar_choice

            # if not st.toggle("Test all {}".format(curr_variation), value=True):
            #     curr_var_filter = st.multiselect('Select Variations to test', unique_curr_variation)
            if not st.toggle(f"Test all {curr_variation} variants", value=st.session_state.cache_filter_currvar_togg, key='k_filter_currvar_togg', on_change=curr_variation_filter_changed):
                curr_var_filter = st.multiselect('Select variants to test', unique_curr_variation)
                ## This method here is to keep the selected items in the multiselect box
                ## but it will annoyingly close the selection tab when the selection is changed
                # curr_var_filter = st.multiselect(
                #     label=f'Select {curr_variation} to test', 
                #     options=unique_curr_variation, 
                #     default=st.session_state.cache_filter_currvar_choice if all(item in unique_curr_variation for item in st.session_state.cache_filter_currvar_choice) else [],
                #     key='k_filter_currvar_choice', 
                #     on_change=curr_variation_filter_choice_changed
                # )
                filter_dict[curr_variation] = curr_var_filter
            else:
                if curr_variation in filter_dict:
                    del filter_dict[curr_variation]


            if 'cache_filter_pinvar_togg' not in st.session_state:
                st.session_state.cache_filter_pinvar_togg = True
            def pin_variation_filter_changed():
                st.session_state.cache_filter_pinvar_togg = st.session_state.k_filter_pinvar_togg

            if 'cache_filter_pinvar_choice' not in st.session_state:
                st.session_state.cache_filter_pinvar_choice = []
            def pin_variation_filter_choice_changed():
                st.session_state.cache_filter_pinvar_choice = st.session_state.k_filter_pinvar_choice

            # if not st.toggle("Test all {}".format(pin_variation), value=True):
                # pin_var_filter = st.multiselect('Select Variations to test', unique_pin_variation)
            if not st.toggle(f"Test all {pin_variation} variants", value=st.session_state.cache_filter_pinvar_togg, key='k_filter_pinvar_togg', on_change=pin_variation_filter_changed):
                pin_var_filter = st.multiselect('Select variants to test', unique_pin_variation)
                ## same as above
                # pin_var_filter = st.multiselect(
                #     label=f'Select {pin_variation} to test', 
                #     options=unique_pin_variation, 
                #     default=st.session_state.cache_filter_pinvar_choice if all(item in unique_pin_variation for item in st.session_state.cache_filter_pinvar_choice) else [],
                #     key='k_filter_pinvar_choice', 
                #     on_change=pin_variation_filter_choice_changed
                # )
                filter_dict[pin_variation] = pin_var_filter
            else:
                if pin_variation in filter_dict:
                    del filter_dict[pin_variation]
            
            # TODO: ADD filters to filter out the absolute block index that do not need to be tested
            # e.g. for the conor structures that has been placed outside the wafer
            # may be enter rows & columns then all the intersections will ignored e.g. enter row 1,2,3; and column 4,5,6 will ignore (1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)
            
            if st.button("Apply Filter", type='primary', use_container_width=True):
                # filtered_df = df_filter(df, st.session_state.filter_dict)
                st.session_state.filter_dict = filter_dict
            col1, col2 = st.columns([1,4])
            with col1:
                dies_to_test = st.session_state.filter_dict.get('die', 'ALL')
                blocks_to_test = st.session_state.filter_dict.get('block_idx_in_die', 'ALL')
                curr_var_to_test = st.session_state.filter_dict.get(curr_variation, 'ALL')
                pin_var_to_test = st.session_state.filter_dict.get(pin_variation, 'ALL')


                st.write("**Applied Filter Settings:**")
                st.write(f"**Dies to test:** {dies_to_test}")
                st.write(f"**In die blocks to test:** {blocks_to_test}")
                st.write(f"**{curr_variation} to test:** {curr_var_to_test}")
                st.write(f"**{pin_variation} to test:** {pin_var_to_test}")
            with col2:
                filtered_df = df_filter(df, st.session_state.filter_dict)
                st.write('**Filtered Data:**')
                st.write(filtered_df)
            st.write("#### Number of DUT to test: ", len(filtered_df))
            if filtered_df.empty:
                st.warning("No DUT to test, please adjust the filter settings")

            # st.write(filtered_df)

    with home_col:
        with st.expander("Select Home Position", expanded=True):
            # initial home position
            if 'home_die' not in st.session_state:
                st.session_state.home_die = df['die'].unique()[0]

            if 'home_block' not in st.session_state:
                st.session_state.home_block = df[df['die'] == st.session_state.home_die]['block_idx_in_die'].unique()[0]

            def home_die_changed():
                st.session_state.home_die = st.session_state.k_home_die
                # when die is changed, block should be updated
                valid_blocks = df[df['die'] == st.session_state.home_die]['block_idx_in_die'].unique()
                st.session_state.home_block = valid_blocks[0] if valid_blocks.size > 0 else None

            def home_block_changed():
                st.session_state.home_block = st.session_state.k_home_block

            # Die selection
            st.selectbox(
                "Select Home Die",
                df['die'].unique(),
                index=list(df['die'].unique()).index(st.session_state.home_die) if st.session_state.home_die in df['die'].unique() else 0,
                key='k_home_die',
                on_change=home_die_changed
            )

            # Block selection (within the selected die)
            valid_blocks = df[df['die'] == st.session_state.k_home_die]['block_idx_in_die'].unique()
            st.selectbox(
                "Select Home Block index within the selected die",
                valid_blocks,
                index=list(valid_blocks).index(st.session_state.home_block) if st.session_state.home_block in valid_blocks else 0,
                key='k_home_block',
                on_change=home_block_changed
            )
            st.write("Current home position: top left DUT of block", st.session_state.k_home_block, "in die", st.session_state.k_home_die)

    if 'cache_repeat_measurements' not in st.session_state:
        st.session_state.cache_repeat_measurements = 1
    def repeat_measurements_changed():
        st.session_state.cache_repeat_measurements = st.session_state.k_repeat_measurements
    
    st.number_input("Repeat measurements", min_value=1, max_value=100, value=st.session_state.cache_repeat_measurements, step=1, key='k_repeat_measurements', on_change=repeat_measurements_changed)

    if 'cache_wafer_idx' not in st.session_state:
        st.session_state.cache_wafer_idx = 0
    def wafer_idx_changed():
        st.session_state.cache_wafer_idx = st.session_state.k_wafer_idx
        
    st.number_input("Wafer index", min_value=0, max_value=100, value=0, step=1, key='k_wafer_idx', on_change=wafer_idx_changed)



if 'measurement_results' not in st.session_state:
    st.session_state.measurement_results = pd.DataFrame({})


##### task control panel section #####
##### sub processing method
if 'process' not in st.session_state:
    st.session_state.process = None
    st.session_state.running = False

script_path = os.path.join(os.getcwd(), "utils/mea_utils.py")


task_control_panel = st.container(border=True)
with task_control_panel:
    st.write("### Control & Status Panel")
    start_button_col, stop_button_col = task_control_panel.columns([1, 1])

with start_button_col:
    start_button = st.button("Initiate Testing", type='primary' if not st.session_state.running else 'secondary', use_container_width=True)
with stop_button_col:
    stop_button = st.button("Terminate", type='primary' if st.session_state.running else 'secondary', use_container_width=True)



if 'subprogress' not in st.session_state:
    st.session_state.subprogress = 0

if 'subresult' not in st.session_state:
    st.session_state.subresult = None

if 'newest_status' not in st.session_state:
    st.session_state.newest_status = None

if 'tested' not in st.session_state:
    st.session_state.tested = 0

if 'tobe_tested' not in st.session_state:
    st.session_state.tobe_tested = 0

if 'subprocess_fnl_stutus' not in st.session_state:
    st.session_state.subprocess_fnl_stutus = None

if 'subprocess_stop_flag' not in st.session_state:
    st.session_state.subprocess_stop_flag = False

if 'measurement_event' not in st.session_state:
    st.session_state.measurement_event = ''

if 'running_task_name' not in st.session_state:
    st.session_state.running_task_name = None


# 当按下按钮时，如果没有其他任务在运行，则启动子程序
if start_button and not st.session_state.running:
    if filtered_df.empty:
        task_control_panel.warning("Failed to initiate, there are no DUT to test, please adjust the filter settings")
    if st.session_state.map_to_issue == {}:
        task_control_panel.warning("Failed to initiate, there are no configurations has been loaded yet")

    if not filtered_df.empty and st.session_state.map_to_issue != {}:

        smu_channel_map = sys_config.SMU_CHANNEL_MAP

        home_idx = df[(df['die'] == st.session_state.k_home_die) & (df['block_idx_in_die'] == st.session_state.k_home_block)]['block'].values[0]
        home_coords = df[df['block'] == home_idx][['x', 'y']].values[0]
        filtered_df = df_filter(df, st.session_state.filter_dict)
        filtered_coords = zip(filtered_df.x, filtered_df.y)
        coords = [-(np.array(coord) - np.array(home_coords)) for coord in filtered_coords] # establishes all structures relative to stage home position
        filtered_df['coords_Href'] = coords
        filtered_df['__curr_variation_object'] = st.session_state.curr_variation
        filtered_df['__pin_variation_object'] = st.session_state.pin_variation
        filtered_df['wafer_idx'] = st.session_state.k_wafer_idx

        pin_cfgs = st.session_state.pin_cfgs

        cfg_map = st.session_state.map_to_issue
        curr_cfgs = st.session_state.design_curr_cfgs

        temp_folder = sys_config.TEMP_FOLDER_PATH
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # filtered_df_path = 'temp/filtered_df.pkl'

        # pin_cfgs_path = 'temp/pin_cfgs.json'
        # cfg_map_path = 'temp/cfg_map.json'
        # curr_cfgs_path = 'temp/curr_cfgs.json'
        # smu_channel_map_path = 'temp/smu_channel_map.json'
        # stop_flag_path = 'temp/stop_flag.txt'

        filtered_df_path = sys_config.TEMP_FILTERED_DF_PATH
        pin_cfgs_path = sys_config.TEMP_PIN_CFGS_PATH
        cfg_map_path = sys_config.TEMP_CFG_MAP_PATH
        curr_cfgs_path = sys_config.TEMP_CURR_CFGS_PATH
        smu_channel_map_path = sys_config.TEMP_SMU_CHANNEL_MAP_PATH
        stop_flag_path = sys_config.TEMP_STOP_FLAG_PATH

        if os.path.exists(stop_flag_path):
            os.remove(stop_flag_path)
        st.session_state.subprocess_stop_flag = False

        # save filtered_df to pickle
        with open(filtered_df_path, 'wb') as f:
            pickle.dump(filtered_df, f)

        # save configurations to json
        with open(pin_cfgs_path, 'w') as f:
            json.dump(pin_cfgs, f)
        with open(cfg_map_path, 'w') as f:
            json.dump(cfg_map, f)
        with open(curr_cfgs_path, 'w') as f:
            json.dump(curr_cfgs, f)
        with open(smu_channel_map_path, 'w') as f:
            json.dump(smu_channel_map, f)

        # initiate the subprocess
        st.session_state.process = subprocess.Popen(
            [
                "python", script_path,
                "--filtered_df", filtered_df_path,
                "--pin_cfgs", pin_cfgs_path,
                "--cfg_map", cfg_map_path,
                "--curr_cfgs", curr_cfgs_path,
                # "--curr_variation", curr_variation,
                # "--pin_variation", pin_variation,
                "--smu_channel_map", smu_channel_map_path,
                # "--nrdgs", "1",
                # "--nplc", "5",
                # "--repeat", "1"
                # "--nrdgs", str(st.session_state.cache_n_readings),
                # "--nplc", str(st.session_state.cache_nplc),
                "--repeat", str(st.session_state.cache_repeat_measurements)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8' # must be set to utf-8 to decode the output
        )

        # reinitialize the variables
        st.session_state.running = True
        st.session_state.tested = 0
        st.session_state.tobe_tested = 0
        st.session_state.running_task_name = sys_config.WAFER_CHARA_TEST_NAME
        st.rerun()

elif start_button and st.session_state.running:
    if st.session_state.running_task_name is not None:
        task_control_panel.info(f"Task {st.session_state.running_task_name} is currently running, please wait...")
    else:
        task_control_panel.info("Task is already running, please wait...")


if stop_button and st.session_state.running:
    stop_flag_path = sys_config.TEMP_STOP_FLAG_PATH

    st.session_state.subprocess_stop_flag = True
    if st.session_state.process is not None:
        with open(stop_flag_path, "w") as f:
            f.write("stop")

elif stop_button and not st.session_state.running:
    st.warning("No task is running")

@st.fragment(run_every='1s')
def status_part():
    with st.container(border=True):
        st.write("#### Real-Time Task Status")
        if st.session_state.subprocess_stop_flag:
            st.info("Stopping the task...")
        
        st.button("Refresh")
        log_placeholder = st.empty()  # placeholder for the log output
        if not st.session_state.running:
            st.session_state.subprogress = 0

        result_status_placeholder = st.empty()  # placeholder for the result status


        log_placeholder.write(st.session_state.newest_status)
        st.progress(st.session_state.subprogress/100)
        result_status_placeholder.write(st.session_state.subresult)
    if st.session_state.running_task_name == sys_config.WAFER_CHARA_TEST_NAME or st.session_state.running_task_name == None:
        if st.session_state.running:
            st.session_state.subresult = f"Task Status: Task running..."

        # if subprocess is running, read the output line by line
        if st.session_state.running and st.session_state.process is not None:
            process = st.session_state.process
            # for line in process.stdout:
            if process.poll() is None:   
                # print([line for line in process.stdout])
                line = process.stdout.readline()
                
                # real-time log output
                st.session_state.newest_status = line.strip()

                # decode the output to get the progress
                match = re.search(r"(\d+)/(\d+) structure tested", line)
                if match:
                    st.session_state.tested = int(match.group(1))       # structure tested
                    st.session_state.tobe_tested = int(match.group(2))        # total structure to be tested
                    st.session_state.subprogress = int(st.session_state.tested / st.session_state.tobe_tested * 100)  # progress percentage
                    # print(st.session_state.subprogress)

            # check if the process has completed
            if process.poll() is not None:  # if the poll() method returns a value, the process has completed
                ## test line
                # print(f"{process.poll()} is not None")
                # print('process.poll() is not None')
                if process.returncode != 0: # problem occurred
                    # result_status_placeholder.error(f"Command failed with return code {process.returncode}")
                    st.session_state.subresult = f"Task Status: Task failed with return code {process.returncode}, log:{[line.strip() for line in process.stdout]}"
                    st.session_state.subprocess_fnl_stutus = 'failed'
                    # print(f"Task failed with return code {process.returncode}, log:{[line.strip() for line in process.stdout]}")
                else: # completed successfully
                    if st.session_state.subprocess_stop_flag:
                        st.warning("Task terminated by user")
                        st.session_state.subresult = "Task Status: Task terminated by user"
                    else:
                        result_status_placeholder.success("Task completed successfully")
                        st.session_state.subresult = "Task Status: Task completed successfully"
                    st.session_state.subprocess_fnl_stutus = 'success' 
                    # # st.session_state.subprocess_fnl_stutus = stu['exception_occurred']
                    # print('Command completed successfully')

                # reset the process and running flag
                st.session_state.running = False
                st.session_state.process = None
                st.session_state.subprocess_stop_flag = False
                st.session_state.running_task_name = None
                st.rerun()        
with task_control_panel:
    status_part()

##### Measurements/Event Results #####
# Fetch the most recent task result
status_file = sys_config.STATUS_FILE_PATH
with st.container(border=True):
    st.write('### Most Recent Task')
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as file:
                stu = json.load(file)
            st.write(stu)
            if stu['exception_occurred']:
                st.warning("Exception occurred during the measurement, message: {}".format(stu['exception_occurred']))
            result_file_path = stu['file_path']
            st.session_state.measurement_event = stu['event_name']
            st.session_state.measurement_results = pd.read_csv(result_file_path)
        except Exception as e:
            st.session_state.measurement_results = pd.DataFrame({})
            if isinstance(e, FileNotFoundError):
                st.error(f"failed to retrive measurement result. Error message: {e}")
            elif isinstance(e, pd.errors.EmptyDataError):
                st.warning(f"No data found in the result file.")
            else:
                st.error(f"Error occurred when reading the result file. Error message: {e}")
        st.write(st.session_state.measurement_results)
