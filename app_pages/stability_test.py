import streamlit as st
import numpy as np
import pandas as pd
import json
import re

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import subprocess

import config as sys_config

import pickle
import copy


from utils.general_utils import convert_string_to_list, dropout_columns

st.title("Stability Test")

###### auto mode ##############
if 'cache_stb_manual_mode' not in st.session_state:
    st.session_state.cache_stb_manual_mode = False

def stb_save_manual_mode():
    st.session_state.cache_stb_manual_mode = st.session_state.k_stb_manual_mode

if not st.toggle(
    'manual mode', 
    key='k_stb_manual_mode', 
    value=st.session_state.cache_stb_manual_mode, 
    on_change=stb_save_manual_mode
):

    st.write("Start with an existing design")

    ## select wafer design
    # folder_path = 'data_collection\wafer_structure_files'
    folder_path = sys_config.WAFER_STRUCTURE_FOLDER_PATH
    file_names = os.listdir(folder_path)

    csv_files = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
    design_names = [f.replace('_wafer_structure.csv', '') for f in csv_files]

    if 'cache_stb_design' not in st.session_state:
        st.session_state.cache_stb_design = design_names[0]

    def stb_save_design():
        st.session_state.cache_stb_design = st.session_state.k_stb_wafer_design

    st.selectbox(
        label='Select Wafer Design', 
        options=design_names, 
        index=design_names.index(st.session_state.cache_stb_design) if st.session_state.cache_stb_design in design_names else 0,
        key='k_stb_wafer_design', 
        on_change=stb_save_design
    )
    df = pd.read_csv(os.path.join(folder_path, f"{st.session_state.k_stb_wafer_design}_wafer_structure.csv"))


    variation_df = df.copy()
    variation_df = dropout_columns(variation_df, ['wafer', 'design', 'x', 'y', 'block', 'die', 'block_row', 'block_column'], inplace=False)
    
    
    ###### structure locator ########
    with st.expander("Structure Locator", expanded=True):
        st.write("### Structure Locator")
        variation_col, die_block_col = st.columns([3,1])
        with variation_col:
            ### select variation objects
            variation_container = st.container(border=True)
            variation_container.write("### Select variation objects")
            curr_variation_col, pin_variation_col = variation_container.columns(2)
            
            if 'cache_stb_curr_variation' not in st.session_state:
                st.session_state.cache_stb_curr_variation = variation_df.columns[0]

            if 'cache_stb_pin_variation' not in st.session_state:
                st.session_state.cache_stb_pin_variation = variation_df.columns[1] if len(variation_df.columns) > 1 else variation_df.columns[0]

            def curr_variation_changed():

                st.session_state.cache_stb_curr_variation = st.session_state.k_stb_curr_variation
                if st.session_state.cache_stb_pin_variation == st.session_state.cache_stb_curr_variation:
                    # if the current variation is the same as the pin variation, change the pin variation to the next available option
                    remaining_options = variation_df.columns.drop(st.session_state.cache_stb_curr_variation)
                    st.session_state.cache_stb_pin_variation = remaining_options[0] if not remaining_options.empty else ""

            def pin_variation_changed():
                st.session_state.cache_stb_pin_variation = st.session_state.k_stb_pin_variation
            with curr_variation_col:
                curr_variation_options = variation_df.columns
                curr_variation = st.selectbox(
                    label='Select current variation object',
                    options=curr_variation_options,
                    index=curr_variation_options.get_loc(st.session_state.cache_stb_curr_variation) if st.session_state.cache_stb_curr_variation in curr_variation_options else 0,
                    key='k_stb_curr_variation',
                    on_change=curr_variation_changed,
                    help="Select the variation that does not requier different pin configurations but requiers different force current configurations, e.g. material, thickness, etc."
                )

            with pin_variation_col:
                pin_variation_options = variation_df.columns.drop(st.session_state.k_stb_curr_variation)
                pin_variation = st.selectbox(
                    label='Select pin variation object',
                    options=pin_variation_options,
                    index=pin_variation_options.get_loc(st.session_state.cache_stb_pin_variation) if st.session_state.cache_stb_pin_variation in pin_variation_options else 0,
                    key='k_stb_pin_variation',
                    on_change=pin_variation_changed,
                    help="Select the variation that requiers different pin configurations, e.g. structure layout, etc."
                )

            ### select variant
            unique_curr_variant = variation_df[str(curr_variation)].unique()
            unique_curr_variant = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in unique_curr_variant]

            unique_pin_variant = variation_df[str(pin_variation)].unique()
            unique_pin_variant = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in unique_pin_variant]


            if 'cache_stb_curr_variant' not in st.session_state:
                st.session_state.cache_stb_curr_variant = unique_curr_variant[0]

            def stb_save_curr_variant():
                st.session_state.cache_stb_curr_variant = st.session_state.k_stb_curr_variant

            with curr_variation_col:
                selected_curr_variant = st.selectbox(
                    label='Select current variant',
                    options=unique_curr_variant,
                    index=unique_curr_variant.index(st.session_state.cache_stb_curr_variant) if st.session_state.cache_stb_curr_variant in unique_curr_variant else 0,
                    key='k_stb_curr_variant',
                    on_change=stb_save_curr_variant
                )

            if 'cache_stb_pin_variant' not in st.session_state:
                st.session_state.cache_stb_pin_variant = unique_pin_variant[0]

            def stb_save_pin_variant():
                st.session_state.cache_stb_pin_variant = st.session_state.k_stb_pin_variant

            with pin_variation_col:
                selected_pin_variant = st.selectbox(
                    label='Select pin variant',
                    options=unique_pin_variant,
                    index=unique_pin_variant.index(st.session_state.cache_stb_pin_variant) if st.session_state.cache_stb_pin_variant in unique_pin_variant else 0,
                    key='k_stb_pin_variant',
                    on_change=stb_save_pin_variant
                )

    
                # die_block_container = st.container()
                # location_col, home_col = die_block_container.columns(2)
                # location_container = location_col.container(border=True)
                # home_container = home_col.container(border=True)
        location_container = die_block_col.container(border=True)
        
        with location_container:
            ### select die
            st.write("### Select die and block")
            if 'cache_stb_die' not in st.session_state:
                st.session_state.cache_stb_die = df['die'].unique()[0]

            def stb_save_die():
                st.session_state.cache_stb_die = st.session_state.k_stb_die

            st.selectbox(
                label='Select die', 
                options=df['die'].unique(), 
                index=list(df['die'].unique()).index(st.session_state.cache_stb_die) if st.session_state.cache_stb_die in df['die'].unique() else 0,
                key='k_stb_die',
                on_change=stb_save_die
            )

            ### select in die block
            if 'cache_stb_block' not in st.session_state:
                st.session_state.cache_stb_block = df['block_idx_in_die'].unique()[0]

            def stb_block_changed():
                st.session_state.cache_stb_block = st.session_state.k_stb_block

            st.selectbox(
                label='select index of block in die',
                options=df['block_idx_in_die'].unique(),
                index=list(df['block_idx_in_die'].unique()).index(st.session_state.cache_stb_block) if st.session_state.cache_stb_block in df['block_idx_in_die'].unique() else 0,
                key='k_stb_block',
                on_change=stb_block_changed
            )


        ### select home die and block
        home_container = st.container(border=True)
        with home_container:
            st.write("### Select home Position")
            if 'cache_stb_home_die' not in st.session_state:
                st.session_state.cache_stb_home_die = df['die'].unique()[0]

            if 'cache_stb_home_block' not in st.session_state:
                st.session_state.cache_stb_home_block = df[df['die'] == st.session_state.cache_stb_home_die]['block_idx_in_die'].unique()[0]

            def stb_home_die_changed():
                st.session_state.cache_stb_home_die = st.session_state.k_stb_home_die
                # when the home die is changed, the home block should be updated to the first valid block in the new die
                valid_blocks = df[df['die'] == st.session_state.cache_stb_home_die]['block_idx_in_die'].unique()
                st.session_state.cache_stb_home_block = valid_blocks[0] if valid_blocks.size > 0 else None

            def stb_home_block_changed():
                st.session_state.cache_stb_home_block = st.session_state.k_stb_home_block

            st.selectbox(
                "Select Home die",
                df['die'].unique(),
                index=list(df['die'].unique()).index(st.session_state.cache_stb_home_die) if st.session_state.cache_stb_home_die in df['die'].unique() else 0,
                key='k_stb_home_die',
                on_change=stb_home_die_changed
            )

            valid_blocks = df[df['die'] == st.session_state.k_stb_home_die]['block_idx_in_die'].unique()
            st.selectbox(
                "Select Home Block index within the selected die",
                valid_blocks,
                index=list(valid_blocks).index(st.session_state.cache_stb_home_block) if st.session_state.cache_stb_home_block in valid_blocks else 0,
                key='k_stb_home_block',
                on_change=stb_home_block_changed
            )
            st.write("Current home position: top left DUT of block", st.session_state.k_stb_home_block, "in die", st.session_state.k_stb_home_die)

    ### display the selected structure
    home_idx = df[(df['die'] == st.session_state.k_stb_home_die) & (df['block_idx_in_die'] == st.session_state.k_stb_home_block)]['block'].values[0]
    home_coords = df[df['block'] == home_idx][['x', 'y']].values[0]
    # home_coords = df[df['block'] == st.session_state.home_idx][['x', 'y']].values[0]
    filtered_df = df[(df['die'] == st.session_state.k_stb_die) & (df['block_idx_in_die'] == st.session_state.k_stb_block) & (df[str(curr_variation)] == st.session_state.k_stb_curr_variant) & (df[str(pin_variation)] == st.session_state.k_stb_pin_variant)].copy()
    filtered_coords = zip(filtered_df.x, filtered_df.y)
    coords = [-(np.array(coord) - np.array(home_coords)) for coord in filtered_coords] # establishes all structures relative to stage home position
    filtered_df['coords_Href'] = coords
    filtered_df['__curr_variation_object'] = st.session_state.k_stb_curr_variation
    filtered_df['__pin_variation_object'] = st.session_state.k_stb_pin_variation

    # st.write(filtered_df)



###### manual mode ##############
else:
    ### fill in the default values
    filtered_df = pd.DataFrame(
        {
            'die': [0],
            'block': [0],
            'x': [0],
            'y': [0],
            'block_row': [0],
            'block_column': [0],
            'in_block_row': [0],
            'in_block_column': [0],
            'block_idx_in_die': [0],
            'design': ['manual_stability_test'],
            'curr_variation': ['manual_curr_setting'],
            'pin_variation': ['manual_pin_setting'],
            'coords_Href': [[0, 0]],
            '__curr_variation_object': ['curr_variation'],
            '__pin_variation_object': ['pin_variation'],
        }
    )
    # st.write(filtered_df)

structure_to_test = filtered_df.copy()
with st.container(border=True):
    st.write("### Structure to test")
    st.write(structure_to_test)

setting_container = st.container(border=True)
with setting_container:
    st.write("### Measurement Configurations")
force_setting_col, measure_setting_col = setting_container.columns(2)

with force_setting_col:
    ### select pin configurations
    # pin_map_file = 'measurement_cfgs\pin_maps.json'
    pin_map_file = sys_config.PIN_MAP_FILE_PATH

    if "pin_cfgs" not in st.session_state:
        with open(pin_map_file, 'r') as file:
            pin_map_file = json.load(file)
            st.session_state.pin_cfgs = pin_map_file


    pin_cfg_options = list(st.session_state.pin_cfgs.keys())

    if 'cache_stb_pin_cfg' not in st.session_state:
        st.session_state.cache_stb_pin_cfg = []

    def stb_save_pin_cfg():
        st.session_state.cache_stb_pin_cfg = st.session_state.k_stb_pin_cfg

    st.multiselect(
        label='Select pin configurations', 
        options=pin_cfg_options,
        default=st.session_state.cache_stb_pin_cfg if all(item in pin_cfg_options for item in st.session_state.cache_stb_pin_cfg) else [],
        key='k_stb_pin_cfg',
        on_change=stb_save_pin_cfg
    )

    ### select force current values
    if 'cache_stb_force_current' not in st.session_state:
        st.session_state.cache_stb_force_current = ''

    def stb_save_force_current():
        st.session_state.cache_stb_force_current = st.session_state.k_stb_force_current

    new_current = st.text_input(f'Enter force current values in Amp', value=st.session_state.cache_stb_force_current, key='k_stb_force_current', help="Enter the force current values separated by commas, e.g. 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1", on_change=stb_save_force_current)
    force_current_list = convert_string_to_list(new_current)
    if any(value >= 1 for value in force_current_list):
        st.warning("High current values detected, please double check make sure the values are in the correct format.")
    st.write(force_current_list)



with measure_setting_col:
    ### select measurement parameters
    if 'cache_stb_nplc' not in st.session_state:
        st.session_state.cache_stb_nplc = 5.0
    def stb_nplc_changed():
        st.session_state.cache_stb_nplc = st.session_state.k_stb_nplc

    # st.number_input("Nplc", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key='k_stb_nplc')
    st.number_input("Nplc", min_value=0.01, max_value=10.0, value=st.session_state.cache_stb_nplc, step=0.01, key='k_stb_nplc', on_change=stb_nplc_changed)

    if 'cache_stb_n_readings' not in st.session_state:
        st.session_state.cache_stb_n_readings = 1
    def stb_n_readings_changed():
        st.session_state.cache_stb_n_readings = st.session_state.k_stb_n_readings

    # st.number_input("N readings", min_value=1, max_value=100, value=10, step=1, key='k_stb_n_readings')
    st.number_input("N readings", min_value=1, max_value=100, value=st.session_state.cache_stb_n_readings, step=1, key='k_stb_n_readings', on_change=stb_n_readings_changed)




    if 'cache_stb_repeat_measurements' not in st.session_state:
        st.session_state.cache_stb_repeat_measurements = 10

    def stb_repeat_measurements_changed():
        st.session_state.cache_stb_repeat_measurements = st.session_state.k_stb_repeat_measurements
    # st.number_input("Repeat measurements", min_value=1, max_value=100, value=1, step=1, key='k_stb_repeat_measurements')
    st.number_input("Repeat measurements for each current setting", min_value=1, max_value=100, value=st.session_state.cache_stb_repeat_measurements, step=1, key='k_stb_repeat_measurements', on_change=stb_repeat_measurements_changed)

    if 'cache_stb_comp_v' not in st.session_state:
        st.session_state.cache_stb_comp_v = 1.0

    def stb_comp_v_changed():
        st.session_state.cache_stb_comp_v = st.session_state.k_stb_comp_v
    st.number_input(
        label = "Compliance Voltage in V", 
        min_value=0.01, 
        max_value=10.0, 
        value=1.0, 
        step=0.01, 
        key='k_stb_comp_v', 
        on_change=stb_comp_v_changed
    )


# fragement to process and show the most recent measurement results
@st.fragment
def show_measurement_results():
    import plotly.graph_objects as go

    with st.expander("Previous Result", expanded=True):
        st.write("### Most Reasent Measurement Results")
        if 'stb_measurement_results' not in st.session_state:
            st.session_state.stb_measurement_results = pd.DataFrame({})

        # status_file = 'temp/stb_stu.json'
        status_file = sys_config.STABILITY_STATUS_FILE_PATH
        if os.path.exists(status_file):
            # if st.session_state.stb_subprocess_fnl_stutus == 'success':
            try:
                with open(status_file, 'r', encoding='utf-8') as file:
                    stu = json.load(file)

                exception_occurred = stu.get('exception_occurred', None)
                # if exception_occurred:
                #     st.warning("Exception occurred during the measurement, message: {}".format(stu['exception_occurred']))
                # result_file_path = stu['file_path']
                result_file_path = stu.get('file_path', None)
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    st.error(f"failed to retrive status file. Error message: {e}")
                else:
                    st.error(f"Error occurred when reading the status file. Error message: {e}")
                stu = {}
            
            try:
                if result_file_path is not None:
                    st.session_state.stb_measurement_event = stu['event_name']
                    st.session_state.stb_measurement_results = pd.read_csv(result_file_path)
            except Exception as e:
                st.session_state.stb_measurement_results = pd.DataFrame({})
                if isinstance(e, FileNotFoundError):
                    st.error(f"failed to retrive measurement result. Error message: {e}")
                elif isinstance(e, pd.errors.EmptyDataError):
                    st.warning(f"No data found in the result file.")
                else:
                    st.error(f"Error occurred when reading the result file. Error message: {e}")

    
        
        if exception_occurred:
            st.warning("Exception occurred during the measurement, message: {}".format(stu['exception_occurred']))
        
        st.write(stu)
        ## measurement results

        result_container = st.container(border=True)
        original_result_col, processed_result_col = result_container.columns(2)
        with original_result_col:
            st.write("#### Retrieved Measurement Results")
            st.write(st.session_state.stb_measurement_results.style.format(precision=7))

        ## plot and process
        
        if not st.session_state.stb_measurement_results.empty:
            
            measurement_results = st.session_state.stb_measurement_results.copy()
            pin_var = st.session_state.stb_measurement_results['__pin_variation_object'].values[0]
            curr_var = st.session_state.stb_measurement_results['__curr_variation_object'].values[0]
            measurement_results = measurement_results.groupby(
                        # ['die', 'block', 'curr_variation', 'pin_variation', 'force_current', 'alias', 'block_row', 'block_column', 'event_counter'],
                        ['die', 'block', pin_var, curr_var, 'force_current', 'alias', 'block_row', 'block_column', 'event_counter'],
                        as_index=False
                    ).agg({'meanV': 'mean', 'meanI': 'mean', 'hit_comp': 'any'})
            measurement_results['r'] = measurement_results['meanV'] / measurement_results['force_current']

            with processed_result_col:
                st.write("#### Process Measurement Results")
                st.select_slider('select the process method', options=['Resistance', 'Sheet_Resistance'], key='k_stb_process_method')
                if st.session_state.k_stb_process_method == 'Resistance':

                    # Grouping the data by current and calculating the mean and standard deviation for Rs
                    grouped_data = measurement_results.groupby('force_current')['r'].agg(['mean', 'std', 'count']).reset_index()
                    grouped_data['sem'] = grouped_data['std'] / np.sqrt(grouped_data['count'])
                    grouped_data['type'] = 'r'
                else:
                    st.selectbox('select the structure type', options=['Greek-Cross', 'Bridge'], key='k_stb_structure_type')
                    if st.session_state.k_stb_structure_type == 'Greek-Cross':
                        # Grouping the data by current and calculating the mean and standard deviation for Rs
                        measurement_results['Rs'] = (
                            np.pi * measurement_results['r'] / np.log(2)
                        )
                    elif st.session_state.k_stb_structure_type == 'Bridge':
                        st.number_input('Enter the designed width of the bridge in um', min_value=0.000000001, max_value=100000.0, value=10.0, step=1.0, key='k_stb_bridge_width')
                        st.number_input('Enter the designed length of the bridge in um', min_value=0.000000001, max_value=100000.0, value=100.0, step=1.0, key='k_stb_bridge_length')
                        measurement_results['Rs'] = (
                            measurement_results['r'] * st.session_state.k_stb_bridge_width / st.session_state.k_stb_bridge_length
                        )
                    grouped_data = measurement_results.groupby('force_current')['Rs'].agg(['mean', 'std', 'count']).reset_index()
                    grouped_data['sem'] = grouped_data['std'] / np.sqrt(grouped_data['count'])
                    grouped_data['type'] = 'Rs'

                st.write(grouped_data.style.format(precision=6))

                data_folder = sys_config.STABILITY_PROCESSED_DATA_FOLDER

                stability_test_name = st.text_input("Enter the name of the stability test", value=f"{stu.get('event_name', 'stability_test')}_{st.session_state.k_stb_process_method}")
                if st.button('Save Processed Results'):
                    if st.session_state.k_stb_manual_mode:
                        design_name_tosave = 'manual'
                    else:
                        design_name_tosave = f"{st.session_state.k_stb_wafer_design}"
                    data_folder = os.path.join(data_folder, design_name_tosave)

                    if stability_test_name.strip() == '':
                        st.warning("Please enter a valid name for the stability test.")
                    else:
                        try:
                            if not os.path.exists(data_folder):
                                os.makedirs(data_folder)
                            file_path = os.path.join(data_folder, f"{stability_test_name}.csv")
                            grouped_data.to_csv(file_path, index=False)
                            st.success(f"Measurement results saved to {file_path}")
                        except Exception as e:
                            st.error(f"Failed to save the measurement results. Error message: {e}")
            # Creating a figure with Plotly
            fig = go.Figure()

            # Adding the Mean Rs with error bars (SEM)
            fig.add_trace(go.Scatter(
                x=grouped_data['force_current'],
                y=grouped_data['mean'],
                error_y=dict(
                    type='data',
                    array=grouped_data['sem'],
                    visible=True
                ),
                mode='lines+markers',
                name='R (Mean with SEM)' if st.session_state.k_stb_process_method == 'Resistance' else 'Sheet Resistance (Mean with SEM)',
                marker=dict(symbol='circle', size=10),
                line=dict(dash='solid')
            ))

            # Adding the Standard Deviation
            fig.add_trace(go.Scatter(
                x=grouped_data['force_current'],
                y=grouped_data['std'],
                mode='lines+markers',
                name='Standard Deviation (S.D.)',
                marker=dict(symbol='square', size=8),
                line=dict(dash='dash')
            ))

            # Setting the layout to match the provided style
            fig.update_layout(
                title=f'Mean {st.session_state.k_stb_process_method} Measurement with Error Bars (SEM) and Standard Deviation',
                xaxis=dict(
                    title='Test Current (A)',
                    type='log'
                ),
                yaxis=dict(
                    # title='Sheet Resistance (Ω/□)',
                    title = 'Resistance (Ω)' if st.session_state.k_stb_process_method == 'Resistance' else 'Sheet Resistance (Ω/□)',
                ),
                yaxis2=dict(
                    # title='Standard Deviation (Ω/□)',
                    title='Standard Deviation (Ω)' if st.session_state.k_stb_process_method == 'Resistance' else 'Standard Deviation (Ω/□)',
                    overlaying='y',
                    side='right'
                ),
                # legend=dict(x=0.01, y=0.99),
                legend=dict(
                    x=1.02,  # right
                    y=1,  # top
                    xanchor='left',  # anchor to the right
                    yanchor='top',  # anchor to the top
                ),
                template='plotly_white'
            )
            st.plotly_chart(fig)
show_measurement_results()




#initialize the session state variables
if 'process' not in st.session_state:
    st.session_state.process = None
    st.session_state.running = False
    st.session_state.running_task_name = None

script_path = os.path.join(os.getcwd(), "utils/mea_utils.py")

task_control_panel = st.container(border=True)
with task_control_panel:
    st.write("### Control & Status Panel")
start_button_col, stop_button_col = task_control_panel.columns([1, 1])

with start_button_col:
    start_button = st.button("Initiate Testing", type='primary' if not st.session_state.running else 'secondary', use_container_width=True)
with stop_button_col:
    stop_button = st.button("Terminate", type='primary' if st.session_state.running else 'secondary', use_container_width=True)


if 'stb_subprogress' not in st.session_state:
    st.session_state.stb_subprogress = 0

if 'stb_subresult' not in st.session_state:
    st.session_state.stb_subresult = None

if 'stb_newest_status' not in st.session_state:
    st.session_state.stb_newest_status = None

if 'stb_tested' not in st.session_state:
    st.session_state.stb_tested = 0

if 'stb_tobe_tested' not in st.session_state:
    st.session_state.stb_tobe_tested = 0

if 'stb_subprocess_fnl_stutus' not in st.session_state:
    st.session_state.stb_subprocess_fnl_stutus = None

if 'stb_subprocess_stop_flag' not in st.session_state:
    st.session_state.stb_subprocess_stop_flag = False

if 'stb_measurement_event' not in st.session_state:
    st.session_state.stb_measurement_event = ''

if 'running_task_name' not in st.session_state:
    st.session_state.running_task_name = None


if start_button and not st.session_state.running:

    smu_channel_map = sys_config.SMU_CHANNEL_MAP
    
    if not st.session_state.k_stb_manual_mode:
        curr_cfgs = {'stability_currs': {selected_curr_variant: force_current_list}}
        cfg_map = {
            selected_pin_variant: 
                {
                    "0": {
                        "alias": selected_pin_variant, 
                        "pin_cfgs": st.session_state.k_stb_pin_cfg, 
                        "curr_cfg": 'stability_currs', 
                        "comp_v": st.session_state.k_stb_comp_v, 
                        "nplc": st.session_state.k_stb_nplc, 
                        "nrdgs": st.session_state.k_stb_n_readings
                    }
            }
        }
        curr_variation = st.session_state.k_stb_curr_variation
        pin_variation = st.session_state.k_stb_pin_variation
        manual_mode = False


    else:
        curr_cfgs = {'stability_currs': {'manual_curr_setting': force_current_list}}
        cfg_map = {
            'manual_pin_setting': {
                "0": {
                    "alias": 'manual_pin_setting', 
                    "pin_cfgs": st.session_state.k_stb_pin_cfg, 
                    "curr_cfg": 'stability_currs', 
                    "comp_v": st.session_state.k_stb_comp_v, 
                    "nplc": st.session_state.k_stb_nplc, 
                    "nrdgs": st.session_state.k_stb_n_readings
                }
            }
        }
        curr_variation = 'curr_variation'
        pin_variation = 'pin_variation'
        manual_mode = True
        st.write('manual mode activated')


    pin_cfgs = st.session_state.pin_cfgs

    temp_folder = sys_config.TEMP_FOLDER_PATH
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    filtered_df_path = sys_config.TEMP_FILTERED_DF_PATH
    pin_cfgs_path = sys_config.TEMP_PIN_CFGS_PATH
    cfg_map_path = sys_config.TEMP_CFG_MAP_PATH
    curr_cfgs_path = sys_config.TEMP_CURR_CFGS_PATH
    smu_channel_map_path = sys_config.TEMP_SMU_CHANNEL_MAP_PATH
    stop_flag_path = sys_config.TEMP_STOP_FLAG_PATH


    if os.path.exists(stop_flag_path):
        os.remove(stop_flag_path)
    st.session_state.stb_subprocess_stop_flag = False

    # save as pickled data frame
    with open(filtered_df_path, 'wb') as f:
        pickle.dump(filtered_df, f)

    # save other configurations as json files
    with open(pin_cfgs_path, 'w') as f:
        json.dump(pin_cfgs, f)
    with open(cfg_map_path, 'w') as f:
        json.dump(cfg_map, f)
    with open(curr_cfgs_path, 'w') as f:
        json.dump(curr_cfgs, f)
    with open(smu_channel_map_path, 'w') as f:
        json.dump(smu_channel_map, f)

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
            # "--nrdgs", str(st.session_state.k_stb_n_readings),
            # "--nplc", str(st.session_state.k_stb_nplc),
            "--repeat", str(st.session_state.k_stb_repeat_measurements),
            "--stability_test", str(True),
            '--manual_mode', str(manual_mode)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8'
    )


    st.session_state.running = True
    # st.session_state.running_task_name = 'Stability_Test'
    st.session_state.running_task_name = sys_config.STABILITY_TEST_NAME
    st.session_state.stb_tested = 0
    st.session_state.stb_tobe_tested = 0
    st.rerun()

elif start_button and st.session_state.running:
    if st.session_state.running_task_name is not None:
        task_control_panel.info(f"{st.session_state.running_task_name} is currently running, please wait...")
    else:
        task_control_panel.info("Task is already running, please wait...")




if stop_button and st.session_state.running:
    # stop_flag_path = 'temp/stop_flag.txt'
    stop_flag_path = sys_config.TEMP_STOP_FLAG_PATH

    st.session_state.stb_subprocess_stop_flag = True
    if st.session_state.process is not None:
        with open(stop_flag_path, "w") as f:
            f.write("stop") 

elif stop_button and not st.session_state.running:
    st.warning("No task is running")

@st.fragment(run_every='2s')
def status_part():
    
    if st.session_state.stb_subprocess_stop_flag:
        st.info("Stopping the task...")
    with st.container(border=True):


        st.write("#### Real-Time Task Status")

        st.button("Refresh")
        log_placeholder = st.empty()  # placeholder for the log
        if not st.session_state.running:
            st.session_state.stb_subprogress = 0
        # progress_bar_placeholder = st.empty()  # placeholder for the progress bar
        result_status_placeholder = st.empty()  # placeholder for the result status


        log_placeholder.write(st.session_state.stb_newest_status)
        # st.session_state.subprocess_progress_bar.progress(st.session_state.stb_subprogress/100)
        st.progress(st.session_state.stb_subprogress/100)
        result_status_placeholder.write(st.session_state.stb_subresult)

    if st.session_state.running_task_name == sys_config.STABILITY_TEST_NAME or st.session_state.running_task_name == None:
        if st.session_state.running:
            st.session_state.stb_subresult = f"Task Status: Task running..."



        # if subprocess is running, check the status of the process
        if st.session_state.running and st.session_state.process is not None:
            
            process = st.session_state.process
            # for line in process.stdout:
            if process.poll() is None:   
                # print([line for line in process.stdout])
                line = process.stdout.readline()
                
                # real-time update the status
                st.session_state.stb_newest_status = line.strip()

                # decode the output to get the progress
                match = re.search(r"Executing stability measurement (\d+)/(\d+)", line)
                # Executing stability measurement (\d+)/(\d+)
                if match:
                    st.session_state.stb_tested = int(match.group(1))       # number of structures tested
                    st.session_state.stb_tobe_tested = int(match.group(2))        # total number of structures to be tested
                    st.session_state.stb_subprogress = int(st.session_state.stb_tested / st.session_state.stb_tobe_tested * 100)  # percentage of the progress
                    # print(st.session_state.stb_subprogress)

            # check if the process has completed
            if process.poll() is not None:  # if the poll() method returns a value, the process has completed
                # print(f"{process.poll()} is not None")
                # print('process.poll() is not None')
                if process.returncode != 0: # problem occurred
                    # result_status_placeholder.error(f"Command failed with return code {process.returncode}")
                    st.session_state.stb_subresult = f"Task Status: Task failed with return code {process.returncode}, log:{[line.strip() for line in process.stdout]}"
                    st.session_state.stb_subprocess_fnl_stutus = 'failed'
                    # print(f"Task failed with return code {process.returncode}, log:{[line.strip() for line in process.stdout]}")
                else: # completed successfully
                    if st.session_state.stb_subprocess_stop_flag:
                        st.warning("Task terminated by user")
                        st.session_state.stb_subresult = "Task Status: Task terminated by user"
                    else:
                        result_status_placeholder.success("Task completed successfully")
                        st.session_state.stb_subresult = "Task Status: Task completed successfully"
                    st.session_state.stb_subprocess_fnl_stutus = 'success' 
                    # # st.session_state.stb_subprocess_fnl_stutus = stu['exception_occurred']
                    # print('Command completed successfully')
                # line = process.stdout.readline()
                line = [line.strip() for line in process.stdout]
                line = line[-1] if line else ''
                
                # real-time update the status
                st.session_state.stb_newest_status = line.strip()

                # reset the process and running status
                st.session_state.running = False
                st.session_state.process = None
                st.session_state.stb_subprocess_stop_flag = False
                st.session_state.running_task_name = None
                st.rerun()        
with task_control_panel:
    status_part()