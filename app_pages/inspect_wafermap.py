import streamlit as st
from streamlit import session_state as ss

import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc


import numpy as np
import json

import config as sys_config



from utils.general_utils import save_to_json


def draw_heat_hist(df: pd.DataFrame, obj: str = 'Rs', curr_variation: str = 'material', pin_variation: str = 'structure', color_scale_max: float = None, color_scale_min: float = None, number_of_bins: int = None, hist_upper_bound: float = None, hist_lower_bound: float = None):
  # Create a heatmap with Plotly

    custom_colorscale = [
        (0.0, "red"),  # Start at red for the lowest value
        (0.000001, "white"),  # Middle neutral (close to zero)
        (0.000002, pc.get_colorscale("Viridis")[0][1]),  # Transition to Viridis at positive values
        *[(i[0], i[1]) for i in pc.get_colorscale("Viridis")[1:]]  # Keep the rest of Viridis

    ]

    if color_scale_max is None:
        zmax = df[obj].quantile(0.95)
    else:
        zmax = color_scale_max
    
    if color_scale_min is None:
        zmin = 0
    else:
        zmin = color_scale_min

    if obj not in df.columns:
        raise ValueError(f"Column '{obj}' not found in the DataFrame")
    if obj == 'Rs':
        obj_title = 'Sheet Resistance (Rs) [Ohm/sq]'
    elif obj == 'linewidth':
        obj_title = 'Electrical Linewidth (W) [um]'


    max_col = df["block_column"].max()
    min_col = df["block_column"].min()
    max_row = df["block_row"].max()
    min_row = df["block_row"].min()

    # Create a full grid of all possible row and column combinations

    full_index = pd.MultiIndex.from_product(
        [range(min_row, max_row + 1), range(min_col, max_col + 1)],
        names=["block_row", "block_column"]
    )

    # Reindex the DataFrame to include all combinations, filling missing values with None
    df_filled = df.set_index(["block_row", "block_column"]).reindex(full_index).reset_index()

    # df_filled["Rs"] = df_filled["Rs"].where(pd.notnull(df_filled["Rs"]), None)
    df_filled[obj] = df_filled[obj].where(pd.notnull(df_filled[obj]), None)
    df_filled["z_highlight"] = df_filled[obj].apply(lambda x: -1 if x is not None and x < 0 else x)

    obj_disp_str = f"{obj_title}"+": %{customdata[0]}<br>"
    additional_lw_disp_str = "GC_Rs [Ohm/sq]: %{customdata[8]}<br> Bridge_r [Ohm]: %{customdata[9]}<br>"

    if obj != 'linewidth':
        custom_list = [obj, "die", "block_idx_in_die", "block", curr_variation, pin_variation, "alias", "hit_comp"]
    elif obj == 'linewidth':
        custom_list = [obj, "die", "block_idx_in_die", "block", curr_variation, pin_variation, "alias", "hit_comp", "GC_Rs", "Bridge_r"]
        obj_disp_str += additional_lw_disp_str
    
    curr_variation_disp_str = f"{curr_variation}"+": %{customdata[4]}<br>"
    pin_variation_disp_str = f"{pin_variation}"+": %{customdata[5]}<br>"

    df = df_filled.copy()
    heatmap = go.Heatmap(
        x=df["block_column"],
        y=df["block_row"],
        # z=df["Rs"],
        # z=df[obj],
        z=df["z_highlight"],
        # colorscale="Viridis",
        colorscale=custom_colorscale,
        # zmin=0,  # Set minimum value for the colorbar to 0
        zmin=zmin,  # Set minimum value for the colorbar to 0
        # zmax=upper_bound_dist,  # Set maximum value for the colorbar to the maximum value in the data
        zmax=zmax,  # Set maximum value for the colorbar to the maximum value in the data
        # colorbar=dict(title="Rs"),
        colorbar=dict(title=obj_title),
        hovertemplate=(
            "Block Column: %{x}<br>"
            "Block Row: %{y}<br>"
            f'{obj_disp_str}'
            "Die: %{customdata[1]}<br>"
            "Block Index in Die: %{customdata[2]}<br>"
            "Abs Block Idx: %{customdata[3]}<br>"
            f'{curr_variation_disp_str}'
            f'{pin_variation_disp_str}'
            "Alias: %{customdata[6]}<br>"
            "Hit Complianace: %{customdata[7]}<br>"
            "<extra></extra>"
        ),
        # customdata=df[[obj, "die", "block_idx_in_die", "block", "material", "structure", "alias", "hit_comp"]].values,
        customdata=df[custom_list].values,
    )

    # Create the figure
    fig = go.Figure(data=heatmap)

    # Update layout
    fig.update_layout(
        width=1000,
        height=1000,
        # title="Heatmap of Rs by Block Column and Block Row",
        title=f"Heatmap of {obj_title} by Block Column and Block Row",
        xaxis_title="Block Column",
        yaxis_title="Block Row",
        # yaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed", scaleanchor="x"),  # Lock aspect ratio to make cells square
        xaxis=dict(scaleanchor="y"),  # Lock aspect ratio to make cells square
        # hovermode="closest",
        template="plotly",
    )

    # Create a histogram of the given values
    # Dynamically calculate the number of bins using the Freedman-Diaconis rule
    def calculate_bins(data):
        data = data.dropna()  # Remove NaN values
        if not data.empty:
            q75, q25 = np.percentile(data, [75 ,25])
            iqr = q75 - q25  # Interquartile range
            bin_width = 3 * iqr * (len(data) ** (-1/3))  # Bin width
            # bin_width = 10
            if bin_width > 0:
                n_bins = int((data.max() - data.min()) / bin_width)
                return max(1, n_bins)  # Ensure at least 1 bin
        else:
            return 10
    
    # filter for outliers
    if hist_upper_bound is None:
        upper_bound = df_filled[obj].quantile(0.95)
    else:
        upper_bound = hist_upper_bound
    
    if hist_lower_bound is None:
        lower_bound = 0
    else:
        lower_bound = hist_lower_bound

    df_filled.dropna(subset=[obj], inplace=True)
    if not hist_upper_bound is None or not hist_lower_bound is None:
        filtered_df = df_filled[df_filled[obj].between(lower_bound, upper_bound, inclusive="both")]    
        df_filled = filtered_df.copy()
    
    if number_of_bins is None:
        n_bins = calculate_bins(df_filled[obj])
    else:
        n_bins = number_of_bins
    # n_bins = calculate_bins(df_filled[obj])

    fig_hist = px.histogram(
        df_filled,
        x=obj,
        nbins=n_bins,  # Specify the number of bins
        # title="Histogram of Rs Values",
        title=f"Histogram of {obj_title}",
        # labels={"Rs": "Resistance (Rs)"},
        labels={obj: obj_title},
        template="plotly",
        color_discrete_sequence=["#636EFA"]  # Use a consistent color for the histogram
    )

    # Update layout for better readability
    fig_hist.update_layout(
        # xaxis_title="Resistance (Rs)",
        width=900,
        height=700,
        xaxis_title=obj_title,
        yaxis_title="Frequency",
    )
    # st.write(df) # testline
    return fig, fig_hist

st.title('Wafer Map Report')


## init key session state variables
if 'alias_lw_map' not in ss:
    try:
        with open(sys_config.ALIAS_LW_MAP_FILE_PATH) as json_file:
            ss.alias_lw_map = json.load(json_file)
    except Exception as e:
        ss.alias_lw_map = {}

if 'measurements_to_process' not in ss:
    # ss.measurements_to_process = pd.DataFrame({})
    ss.measurements_to_process = None

if 'measurement_results' not in ss:
    ss.measurement_results = pd.DataFrame({})

if 'rs_df' not in ss:
    ss.rs_df = pd.DataFrame({})

if 'linewidth_df' not in ss:
    ss.linewidth_df = pd.DataFrame({})

if 'measurement_event' not in st.session_state:
    st.session_state.measurement_event = ''

######## Load New Event ########
with st.expander('Load New Event', expanded=True):
    event_select_cols = st.columns(2)
    with event_select_cols[0]:
        st.write('### Continue with the most recent event:')
        # if not ss.get('measurement_results', None).empty:
        if not ss.get('measurement_results', None) is None and not ss.measurement_results.empty:
            st.write(ss.measurement_results)
            if st.button('Continue'):
                ss.measurements_to_process = ss.measurement_results.copy()
                ss.vis_event_name = ss.measurement_event
        else:
            st.write('No recent event or recent event is empty')

    with event_select_cols[1]:
        st.write('### Or Load a previous event:')

        measurement_result_folders = os.listdir(sys_config.GROUPED_MEASUREMENTS_FOLDER)
        if 'vis_selected_design' not in ss:
            ss.vis_selected_design = measurement_result_folders[0]

        def vis_design_changed():
            ss.vis_selected_design = ss.k_vis_selected_design
            del ss.vis_selected_event

        st.selectbox(
            label='Select the design', 
            options=measurement_result_folders, 
            index=measurement_result_folders.index(ss.vis_selected_design),
            key='k_vis_selected_design',
            on_change=vis_design_changed
        )
        # selected_design = ss.vis_selected_design
        

        event_list = os.listdir(f'{sys_config.GROUPED_MEASUREMENTS_FOLDER}/{ss.vis_selected_design}')
        sorted_event_list = sorted(event_list, reverse=True)
        if 'vis_selected_event' not in ss:
            ss.vis_selected_event = sorted_event_list[0]
        
        def vis_event_changed():
            ss.vis_selected_event = ss.k_vis_selected_event

        st.selectbox(
            label='Select the event record',
            options=sorted_event_list,
            index=sorted_event_list.index(ss.vis_selected_event),
            key='k_vis_selected_event',
            on_change=vis_event_changed
        ) 

        selected_event_path = f'{sys_config.GROUPED_MEASUREMENTS_FOLDER}/{ss.vis_selected_design}/{ss.vis_selected_event}'

        if st.button('Load'):
            try:
                ss.measurements_to_process = pd.read_csv(selected_event_path)
                ss.vis_event_name = ss.vis_selected_event
            except pd.errors.EmptyDataError:
                st.warning('No data can be processed in the selected event')
            
######## End Load New Event ########


######## View in Progress Event ########
if 'vis_event_name' not in ss:
    ss.vis_event_name = ''

if 'cache_vis_event_name' not in ss:
    ss.cache_vis_event_name = ''

if ss.cache_vis_event_name != ss.vis_event_name:
    ss.cache_vis_event_name = ss.vis_event_name
    ss.rs_df = pd.DataFrame({})
    ss.linewidth_df = pd.DataFrame({})

if not ss.measurements_to_process is None:

    if ss.measurements_to_process.empty:
        st.warning('No data to process')
    else:
        if ss.vis_event_name.endswith('_results.csv'):
            # ss.vis_event_name = ss.vis_event_name[:-12]
            processed_event_name = ss.vis_event_name[:-12]
        else:
            # ss.vis_event_name = ss.vis_event_name
            processed_event_name = ss.vis_event_name
        # design_name = ss.vis_event_name.rsplit('_', 4)[0]
        design_name = processed_event_name.rsplit('_', 4)[0]
        if '__curr_variation_object' in ss.measurements_to_process.columns:
            curr_variation = ss.measurements_to_process['__curr_variation_object'].iloc[0]
        elif 'material' in ss.measurements_to_process.columns:
            curr_variation = 'material'
        else:
            st.warning('No variation object found in the data, please check the data or try another event')
            st.stop()
        
        if '__pin_variation_object' in ss.measurements_to_process.columns:
            pin_variation = ss.measurements_to_process['__pin_variation_object'].iloc[0]
        elif 'structure' in ss.measurements_to_process.columns:
            pin_variation = 'structure'
        else:
            st.warning('No pin variation object found in the data, please check the data or try another event')
            st.stop()

        if 'wafer_idx' in ss.measurements_to_process.columns:
            wafer_idx = ss.measurements_to_process['wafer_idx'].iloc[0]
        else:
            ss.measurements_to_process['wafer_idx'] = 'unknown'
            wafer_idx = 'unknown'

        # new update to match the block index in die if not found
        if 'block_idx_in_die' not in ss.measurements_to_process.columns:
            try:
                folder_path = sys_config.WAFER_STRUCTURE_FOLDER_PATH
                block_structure = pd.read_csv(f'{folder_path}/{design_name}_wafer_structure.csv')
                filtered_block_structure = block_structure[block_structure['block'].isin(ss.measurements_to_process['block'])]  # save only the blocks that are tested
                ss.measurements_to_process = pd.merge(ss.measurements_to_process, filtered_block_structure[['block', 'block_idx_in_die']], on='block', how='left')
            except Exception as e:
                st.error(f'Error: {e}')
                # st.stop()
                ss.measurements_to_process['block_idx_in_die'] = None
        

        
        with st.container(border=True):
            st.write('## Loaded Event')
            col1, col2 = st.columns([1,3])
        with col1:
            # with st.container(border=True):
            st.write('### General information')
            st.write(f'**loaded event**: {ss.vis_event_name}')
            st.write(f'**loaded design**: {design_name}')
            # st.write(f'loaded design:{ss.vis_event_name.rsplit('_', 4)[0]}')
            # st.write(f'loaded design:{ss.vis_event_name.rsplit('_', 4)[1]}')
            st.write(f'**loaded wafer**: {wafer_idx}')
            st.write(f'**tested {pin_variation}**: {ss.measurements_to_process["alias"].unique()}')
            st.write(f'**tested {curr_variation}**: {ss.measurements_to_process[curr_variation].unique()}')
            st.write(f'**tested dies**: {ss.measurements_to_process["die"].unique()}')
            st.write(f'**Number of blocks tested**: {ss.measurements_to_process["block"].nunique()}')
                # st.write(f'Number of structures tested: {}')
        with col2:
            # with st.container(border=True):
            st.write('### Loaded Measurements')
            st.write(ss.measurements_to_process)    

        if design_name not in ss.alias_lw_map:
            ss.alias_lw_map[design_name] = {}
######## View in Progress Event End ########

######## Structure info config ########
        if 'cache_vis_bridge_structures' not in st.session_state:
            st.session_state.cache_vis_bridge_structures = []

        if 'cache_vis_gc_structures' not in st.session_state:
            st.session_state.cache_vis_gc_structures = []

        def bridge_structures_changed():
            # refresh cache
            st.session_state.cache_vis_bridge_structures = st.session_state.k_vis_bridge_structures
            # make sure Greek Cross structures do not conflict with Bridge structures
            st.session_state.cache_vis_gc_structures = [
                x for x in st.session_state.cache_vis_gc_structures 
                if x not in st.session_state.cache_vis_bridge_structures
            ]

        def gc_structures_changed():
            # refresh cache
            st.session_state.cache_vis_gc_structures = st.session_state.k_vis_gc_structures
            # make sure Bridge structures do not conflict with Greek Cross structures
            st.session_state.cache_vis_bridge_structures = [
                x for x in st.session_state.cache_vis_bridge_structures 
                if x not in st.session_state.cache_vis_gc_structures
            ]

        # obtain unique structures
        unique_structures = ss.measurements_to_process['alias'].unique()
        with st.expander('Configure Bridge and Greek Cross Structures', expanded=True):
            with st.container(border=True):
                st.write('### Bridge Structures')
                # Bridge structure selection
                bridge_structures = st.multiselect(
                    label='Select the Bridge structures', 
                    options=unique_structures,
                    default=st.session_state.cache_vis_bridge_structures if all([x in unique_structures for x in st.session_state.cache_vis_bridge_structures]) else [],
                    key='k_vis_bridge_structures',
                    on_change=bridge_structures_changed,
                    help="Select the structures corresponding to Bridge."
                )

                # for bridge_structure in bridge_structures:
                for bridge_structure in st.session_state.k_vis_bridge_structures:
                    if bridge_structure not in ss.alias_lw_map[design_name]:
                        ss.alias_lw_map[design_name][bridge_structure] = {}
                        found_width = None
                        found_length = None
                    else: 
                        if 'width' in ss.alias_lw_map[design_name][bridge_structure]:
                            found_width = ss.alias_lw_map[design_name][bridge_structure]['width']
                        else:
                            found_width = None
                        if 'length' in ss.alias_lw_map[design_name][bridge_structure]:
                            found_length = ss.alias_lw_map[design_name][bridge_structure]['length']
                        else:
                            found_length = None
                    
                    st.write(f'#### {bridge_structure}')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.number_input(f'Enter the designed bridge length for {bridge_structure} in micron', 
                                        min_value=0.000001, 
                                        value=found_length if found_length else 100.0,
                                        key=f'{bridge_structure}_length',
                                    )
                    with col2:
                        st.number_input(f'Enter the designed bridge width for {bridge_structure} in micron', 
                                        min_value=0.000001, 
                                        value=found_width if found_width else 10.0,
                                        key=f'{bridge_structure}_width'
                                    )
                    ss.alias_lw_map[design_name][bridge_structure]['length'] = ss.get(f'{bridge_structure}_length', 100.0)
                    ss.alias_lw_map[design_name][bridge_structure]['width'] = ss.get(f'{bridge_structure}_width', 10.0)
            # gc_structures = st.multiselect('Select the Greek Cross structures',  [x for x in ss.measurements_to_process['alias'].unique() if x not in bridge_structures])
            with st.container(border=True):
                st.write('### Greek Cross Structures')
                gc_structures = st.multiselect(
                    label='Select the Greek Cross structures', 
                    options=[x for x in unique_structures if x not in st.session_state.k_vis_bridge_structures],
                    default=st.session_state.cache_vis_gc_structures if all([x in unique_structures for x in st.session_state.cache_vis_gc_structures]) else [],
                    key='k_vis_gc_structures',
                    on_change=gc_structures_changed,
                    help="Select the structures corresponding to Greek Cross."
                )
######## Structure info config END ########

######## PostProcess Data ########
            if st.button('Apply', use_container_width=True, type='primary'):

                # Make a copy of measurement results to avoid changing original data directly
                try:# Save the updated alias_lw_map to a JSON file
                    alias_lw_map_path = sys_config.ALIAS_LW_MAP_FILE_PATH
                    save_to_json(alias_lw_map_path, ss.alias_lw_map)
                except Exception as e: 
                    # not vital to the process, so pass
                    pass
                measurement_results = ss.measurements_to_process.copy()
                # group the measurements for each structure
                measurement_results = measurement_results.groupby(
                    ['die', 'block_idx_in_die', 'block', curr_variation, pin_variation, 'force_current', 'alias', 'block_row', 'block_column'],
                    as_index=False
                ).agg({'meanV': 'mean', 'meanI': 'mean', 'hit_comp': 'any'})
                
                # Calculate resistance (r) for all structures
                measurement_results['r'] = measurement_results['meanV'] / measurement_results['force_current']

                # Initialize Rs column
                measurement_results['Rs'] = None
                
                # Calculate Rs for Greek Cross structures
                for gc_structure in gc_structures:
                    # Greek Cross calculation: Rs = (pi * r) / ln(2)
                    measurement_results.loc[measurement_results['alias'] == gc_structure, 'Rs'] = (
                        np.pi * measurement_results.loc[measurement_results['alias'] == gc_structure, 'r'] / np.log(2)
                    )
                
                # Calculate Rs for Bridge structures
                for bridge_structure in bridge_structures:
                    length = ss.get(f'{bridge_structure}_length', 1.0)
                    width = ss.get(f'{bridge_structure}_width', 1.0)

                    measurement_results.loc[measurement_results['alias'] == bridge_structure, 'Rs'] = (
                        measurement_results.loc[measurement_results['alias'] == bridge_structure, 'r'] * (width / length)
                    )
                    # else:
                    #     st.warning(f'Please enter valid length and width for {bridge_structure}.')
                            # Calculate electrical linewidth (W) for Bridge structures using Greek Cross Rs
                ss.rs_df = measurement_results.copy().dropna(subset=['Rs'])
                # st.write(measurement_results) # testline

                # Calculate electrical linewidth (W) for Bridge structures
                linewidth_results = []
                # average through different current
                measurement_results = measurement_results.groupby(
                    ['die', 'block_idx_in_die', 'block', curr_variation, pin_variation, 'alias', 'block_row', 'block_column'],
                    as_index=False
                ).agg({'Rs': 'mean', 'r': 'mean', 'hit_comp': 'any'})
                for bridge_structure in bridge_structures:
                    # length_in_m = ss.get(f'{bridge_structure}_length', 0.0)*1e-6 # to convert to meters
                    length_in_um = ss.get(f'{bridge_structure}_length', 0.0001)
                    bridge_data = measurement_results[measurement_results['alias'] == bridge_structure]
                    for index, bridge_row in bridge_data.iterrows():
                        # Match Greek Cross with the same material and block
                        greek_cross = measurement_results[(measurement_results[curr_variation] == bridge_row[curr_variation]) &
                                                        (measurement_results['block'] == bridge_row['block']) &
                                                        (measurement_results['alias'].isin(gc_structures))]
                        # st.write(greek_cross)
                        if not greek_cross.empty:
                            # Use the first matching Greek Cross
                            # rs_gc = greek_cross.iloc[0]['Rs']
                            rs_gc = greek_cross['Rs'].mean() # average Rs of all Greek Cross structures
                            gc_comp = greek_cross['hit_comp'].any()  # hit compliance of the Greek Cross structure
                            rl = bridge_row['r']  # Resistance of the Bridge structure
                            
                            # linewidth = rs_gc * length_in_m / rl  # W = (Rs * L) / R_L # for meters
                            linewidth = rs_gc * length_in_um / rl  # W = (Rs * L) / R_L
                            linewidth_results.append({
                                'die': bridge_row['die'],
                                'block_idx_in_die': bridge_row['block_idx_in_die'],
                                'block': bridge_row['block'],
                                curr_variation: bridge_row[curr_variation],
                                pin_variation: bridge_row[pin_variation],
                                'alias': bridge_row['alias'],
                                'linewidth': linewidth,
                                'block_row': bridge_row['block_row'],
                                'block_column': bridge_row['block_column'],
                                'GC_Rs': rs_gc,
                                'Bridge_r': bridge_row['r'],
                                'hit_comp': bridge_row['hit_comp'] or gc_comp # if either of the structures is compliant, the combined structure is compliant
                            })

                # Convert linewidth results to DataFrame
                ss.linewidth_df = pd.DataFrame(linewidth_results)
                # st.write(ss.linewidth_df) # testline
    ##### calculations are doneabove, now display the results
        #initialise the containers for result display
        processed_container = st.container(border=True)
        processed_container.write('## Processed Data')
        selection_col, processed_col = processed_container.columns([2, 3])
        processed_col.empty()


    ##### Select the parametric to view
        process_method = ['Sheet Resistance', 'Electrical Linewideth']
        if 'selected_method' not in ss:
            ss.selected_method = process_method[0]

        def vis_method_changed():
            ss.selected_method = ss.k_selected_method
        with selection_col:
            st.segmented_control(
                label='Select the parametric to view', 
                options=process_method,
                default=ss.selected_method,
                key = 'k_selected_method',
                on_change=vis_method_changed
            )

        # Save the processed data
        save_container = processed_container.container(border=True)
        with save_container:
            data_folder = 'data_collection/processed_data/wafer_maps'
            data_folder = os.path.join(data_folder, design_name)
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            st.text_input(
                'Enter the name of the processed parametric data', 
                value=f'{"Rs" if ss.k_selected_method == "Sheet Resistance" else "Linewidth"}_wafer-{wafer_idx}_{processed_event_name}', 
                key='processed_event_name'
            )
        
        processed_container.write('### Visualization')

        if 'cache_vis_settings' not in ss:
            ss.cache_vis_settings = False
        
        def vis_settings_changed():
            ss.cache_vis_settings = ss.show_vis_settings

        processed_container.toggle(
            'Show Visualization Manual Settings', 
            value=ss.cache_vis_settings,
            key='show_vis_settings',
            on_change=vis_settings_changed
        )
        # initialize the containers for visualization
        wafer_map_col, hist_col = processed_container.columns([1, 1])
        
        if ss.show_vis_settings:
            # for visualization setting
            if 'cache_color_scale_max' not in st.session_state:
                st.session_state.cache_color_scale_max = 5.0  # default max value
            if 'cache_color_scale_min' not in st.session_state:
                st.session_state.cache_color_scale_min = 0.0  # default min value

            def update_color_scale_max():
                # refresh cache
                st.session_state.cache_color_scale_max = st.session_state.color_scale_max
                # make sure the max value is not less than the min value
                if st.session_state.cache_color_scale_min > st.session_state.cache_color_scale_max:
                    st.session_state.cache_color_scale_min = st.session_state.cache_color_scale_max

            def update_color_scale_min():
                # refresh cache
                st.session_state.cache_color_scale_min = st.session_state.color_scale_min
                # make sure the min value is not greater than the max value
                if st.session_state.cache_color_scale_min > st.session_state.cache_color_scale_max:
                    st.session_state.cache_color_scale_max = st.session_state.cache_color_scale_min

            # max value input
            color_scale_max = wafer_map_col.number_input(
                label='color scale max',
                value=st.session_state.cache_color_scale_max,
                key='color_scale_max',
                format='%0.6f',
                on_change=update_color_scale_max
            )

            # use the input max value as the min value
            color_scale_min = wafer_map_col.number_input(
                label='color scale min',
                # min_value=0.0, # not necessary
                value=st.session_state.cache_color_scale_min,
                max_value=st.session_state.cache_color_scale_max,
                key='color_scale_min',
                format='%0.6f',
                on_change=update_color_scale_min,
                help='The minimum value for the color scale, values below this will be colored differently in bright red.'
            )

            hist_col.number_input('number of bins', value=10, key='number_of_bins', format='%d')
            if 'cache_hist_upper_bound' not in st.session_state:
                st.session_state.cache_hist_upper_bound = None
            if 'cache_hist_lower_bound' not in st.session_state:
                st.session_state.cache_hist_lower_bound = None
            
            def update_hist_upper_bound():
                st.session_state.cache_hist_upper_bound = st.session_state.hist_upper_bound
                if st.session_state.cache_hist_upper_bound is not None and st.session_state.cache_hist_lower_bound is not None:
                    if st.session_state.cache_hist_upper_bound < st.session_state.cache_hist_lower_bound:
                        st.session_state.cache_hist_lower_bound = st.session_state.cache_hist_upper_bound
            
            def update_hist_lower_bound():
                st.session_state.cache_hist_lower_bound = st.session_state.hist_lower_bound
                if st.session_state.cache_hist_upper_bound is not None and st.session_state.cache_hist_lower_bound is not None:
                    if st.session_state.cache_hist_upper_bound < st.session_state.cache_hist_lower_bound:
                        st.session_state.cache_hist_upper_bound = st.session_state.cache_hist_lower_bound

            # max value input
            hist_upper_bound = hist_col.number_input(
                label='histogram upper bound', 
                value=st.session_state.cache_hist_upper_bound,
                key='hist_upper_bound',
                format='%0.6f',
                help='The upper bound for the histogram, values above this will be filtered out.',
                on_change=update_hist_upper_bound
            )

            # min value input
            hist_lower_bound = hist_col.number_input(
                label='histogram lower bound', 
                value=st.session_state.cache_hist_lower_bound,
                max_value=hist_upper_bound,
                key='hist_lower_bound',
                format='%0.6f',
                help='The lower bound for the histogram, values below this will be filtered out.',
                on_change=update_hist_lower_bound
            )

        ##### Here is the main display of the processed data
        if ss.selected_method == 'Sheet Resistance':
            # for Rs only
            if not ss.rs_df.empty:
                # st.write(ss.rs_df)
                if selection_col.toggle('average through different current'): # average through different current
                    # st.write('average through different current')
                    if 'averaged_measurement_results' not in ss:
                        ss.averaged_measurement_results = pd.DataFrame({})
                    
                    averaged_measurement_results = ss.rs_df.groupby(['die', 'block_idx_in_die', 'block', curr_variation, pin_variation, 'alias', 'block_row', 'block_column'],
                                                                                                        as_index=False
                                                                                                        ).agg({'hit_comp': 'any', 'r': 'mean', 'Rs': 'mean'})
                    vis_df = averaged_measurement_results
                else:
                    averaged_measurement_results = ss.rs_df
                    # select the current
                    selected_current = selection_col.selectbox('Select the current', averaged_measurement_results['force_current'].unique())
                    vis_df = averaged_measurement_results[averaged_measurement_results['force_current'] == selected_current]
                # select structure type and current variant
                selected_structure = selection_col.selectbox(f'Select the {pin_variation} alias', averaged_measurement_results['alias'].unique())
                selected_material = selection_col.selectbox(f'Select the {curr_variation}', averaged_measurement_results[curr_variation].unique())

                vis_df = vis_df[(vis_df['alias'] == selected_structure) & (vis_df[curr_variation] == selected_material)]
                # display the processed data
                with processed_col: 
                    st.write(vis_df.style.format(precision=7))
                    st.write('#### Number of structures:', len(vis_df))
                # display the visualization
                if not vis_df.empty:
                    if ss.show_vis_settings:
                        fig_heatmap, fig_hist = draw_heat_hist(vis_df, obj = 'Rs', curr_variation=curr_variation, pin_variation=pin_variation,
                                                            color_scale_max=ss.color_scale_max, color_scale_min=ss.color_scale_min, number_of_bins=ss.number_of_bins,
                                                            hist_upper_bound=ss.hist_upper_bound, hist_lower_bound=ss.hist_lower_bound)
                    else:
                        fig_heatmap, fig_hist = draw_heat_hist(vis_df, obj = 'Rs', curr_variation=curr_variation, pin_variation=pin_variation)
                    wafer_map_col.plotly_chart(fig_heatmap)
                    hist_col.plotly_chart(fig_hist)
                else:
                    processed_container.warning('No processed data to view')
                
            else:
                processed_container.warning('No processed data to view')

        # for linewidth only
        elif ss.selected_method == 'Electrical Linewideth':
            if not ss.linewidth_df.empty:
                vis_df = ss.linewidth_df
                selected_structure = selection_col.selectbox(f'Select the {pin_variation} alias', vis_df['alias'].unique())
                selected_material = selection_col.selectbox(f'Select the {curr_variation}', vis_df[curr_variation].unique())
                # filter the data to display
                vis_df = vis_df[(vis_df['alias'] == selected_structure) & (vis_df[curr_variation] == selected_material)]
                processed_col.write(vis_df.style.format(precision=7))
                if not vis_df.empty:
                    if ss.show_vis_settings:
                        fig_heatmap, fig_hist = draw_heat_hist(
                            vis_df, 
                            obj = 'linewidth', 
                            curr_variation=curr_variation, 
                            pin_variation=pin_variation,
                            color_scale_max=ss.color_scale_max, 
                            color_scale_min=ss.color_scale_min, 
                            number_of_bins=ss.number_of_bins,
                            hist_upper_bound=ss.hist_upper_bound,
                            hist_lower_bound=ss.hist_lower_bound
                        )
                    else:
                        fig_heatmap, fig_hist = draw_heat_hist(
                            vis_df, 
                            obj = 'linewidth', 
                            curr_variation=curr_variation, 
                            pin_variation=pin_variation
                        )

                    wafer_map_col.plotly_chart(fig_heatmap)
                    hist_col.plotly_chart(fig_hist)
                else:
                    processed_container.warning('No processed data to view')

            else:
                processed_container.warning('No processed data to view')

        with save_container:
            if st.button(
                label='Save Processed Parametric Data', 
                type='primary', 
                help="""
                Save the processed data as a csv file, 
                this will save all the processed data for selected parametric, 
                no matter the current selection of current, structure or material.
                """
            ):
                name = st.session_state.processed_event_name
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                if name.strip() != '':
                    if not os.path.exists(f'{data_folder}/{name}.csv'):
                        if ss.selected_method == 'Sheet Resistance':
                            if not ss.rs_df.empty:
                                ss.rs_df.to_csv(f'{data_folder}/{name}.csv', index=False)
                                st.success(f'Successfully saved the processed data as {name}.csv')
                            else:
                                st.warning('No processed data to save')
                        elif ss.selected_method == 'Electrical Linewideth':
                            if not ss.linewidth_df.empty:
                                ss.linewidth_df.to_csv(f'{data_folder}/{name}.csv', index=False)
                                st.success(f'Successfully saved the processed data as {name}.csv')
                            else:
                                st.warning('No processed data to save')
                        
                    else:
                        st.warning(f'{name}.csv already exists, please choose another name.')
                else:
                    st.warning('Please enter a valid name for the processed data.')



