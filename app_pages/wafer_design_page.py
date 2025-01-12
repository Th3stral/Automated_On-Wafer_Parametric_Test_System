import streamlit as st
from streamlit import session_state as ss


import pandas as pd

import plotly.graph_objects as go
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


# from data_structure.coor_extraction import get_top_level_cells, flatten_selected, find_pads ,parse_pad_coords

from utils.gds_utils import get_top_level_cells, flatten_selected, find_pads, parse_pad_coords
from utils.general_utils import wafer_structure_to_dataframe, check_blocks_relative_positions, classify_blocks_into_rows_and_columns, reassign, index_blocks_in_die, classify_rows_and_columns_in_blocks, apply_labels_to_blocks, process_for_plotting

import config as sys_config

coord_file_save_folder = sys_config.WAFER_STRUCTURE_FOLDER_PATH
if not os.path.exists(coord_file_save_folder):
    os.makedirs(coord_file_save_folder)



def plot_wafer(df, probe_spacing=240):
    dies_info, blocks_info = process_for_plotting(df)
    # create a plotly figure
    fig = go.Figure()
        # go through each block and draw a rectangle

    ss.progress_bar = st.progress(0, text='plotting the wafer, please wait...')
    for idx, row in blocks_info.iterrows():
        ss.progress_bar.progress(int((idx + 1) / len(blocks_info) * 70), text='plotting the blocks, please wait...')

        x_min, x_max, y_min, y_max = row['x_min'], row['x_max'], row['y_min'], row['y_max']
        _ ="""
        HACK: If the block is a single point, the code will draw a 4x2 rectangle centered at the point
              This is due to the large amount of pad coordinates that would be a burden to handle in the code,
              so this is a quick fix to make the block visible, it is not a perfect solution.
        """
        if x_min == x_max or y_min == y_max:
            width = probe_spacing * 4
            height = probe_spacing * 2
            x_max = x_min + width
            y_min = y_max - height
            one_structure_block = True
        else:
            width = x_max - x_min
            height = y_max - y_min
            one_structure_block = False
        # draw a rectangle for the block
        fig.add_shape(
            type='rect',
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            line=dict(color='blue', width=2)
        )
        # add text to the center of the block
        fig.add_trace(go.Scatter(
            x=[x_min + width / 2],
            y=[y_min + height / 2],
            text=[f"Block {int(row['block'])}"],
            mode='text',
            textposition='middle center',
            textfont=dict(size=10, color='red')
        ))

    # go through each die and draw a rectangle
    for idx, row in dies_info.iterrows():
        ss.progress_bar.progress(int((idx + 1) / len(blocks_info) * 30 + 70), text='plotting the dies, please wait...')
        x_min, x_max, y_min, y_max = row['x_min'], row['x_max'], row['y_min'], row['y_max']
        width = x_max - x_min
        height = y_max - y_min
        if one_structure_block:
            x_max = x_max + probe_spacing * 4
            y_min = y_min - probe_spacing * 2
            width = x_max - x_min
            height = y_max - y_min
        # draw a rectangle for the die
        fig.add_shape(
            type='rect',
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            line=dict(color='green', width=4, dash='dash')
        )
        # add text to the top of the die
        fig.add_trace(go.Scatter(
            x=[x_min + width / 2],
            y=[y_max + 200],
            text=[f"Die {int(row['die'])}"],
            mode='text',
            textposition='top center',
            textfont=dict(size=12, color='green')
        ))

    ss.progress_bar.progress(100, text='plotting completed')
    # update the layout of the figure
    fig.update_layout(
        width=2000,
        height=2000,
        showlegend=False,
        xaxis=dict(title='X coordinate (micron)'),
        yaxis=dict(title='Y coordinate (micron)'),
        title='Positions of Wafer Blocks and Dies'
    )

    return fig

def wafer_plot_generator(df, probe_spacing=240):
    
    dies_info, blocks_info = process_for_plotting(df)
    # create a plotly figure
    fig = go.Figure()
        # go through each block and draw a rectangle

    ss.progress_bar = st.progress(0, text='plotting the wafer, please wait...')
    for idx, row in blocks_info.iterrows():
        ss.progress_bar.progress(int((idx + 1) / len(blocks_info) * 70), text='plotting the blocks, please wait...')

        x_min, x_max, y_min, y_max = row['x_min'], row['x_max'], row['y_min'], row['y_max']
        _ ="""
        HACK: If the block is a single point, the code will draw a 4x2 rectangle centered at the point
              This is due to the large amount of pad coordinates that would be a burden to handle in the code,
              so this is a quick fix to make the block visible, it is not a perfect solution.
        """
        if x_min == x_max or y_min == y_max:
            width = probe_spacing * 4
            height = probe_spacing * 2
            x_max = x_min + width
            y_min = y_max - height
            one_structure_block = True
        else:
            width = x_max - x_min
            height = y_max - y_min
            one_structure_block = False
        # draw a rectangle for the block
        fig.add_shape(
            type='rect',
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            line=dict(color='blue', width=2)
        )
        # add text to the center of the block
        fig.add_trace(go.Scatter(
            x=[x_min + width / 2],
            y=[y_min + height / 2],
            text=[f"Block {int(row['block'])}"],
            mode='text',
            textposition='middle center',
            textfont=dict(size=10, color='red')
        ))
        yield fig

    # go through each die and draw a rectangle
    for idx, row in dies_info.iterrows():
        ss.progress_bar.progress(int((idx + 1) / len(blocks_info) * 30 + 70), text='plotting the dies, please wait...')
        x_min, x_max, y_min, y_max = row['x_min'], row['x_max'], row['y_min'], row['y_max']
        width = x_max - x_min
        height = y_max - y_min
        if one_structure_block:
            x_max = x_max + probe_spacing * 4
            y_min = y_min - probe_spacing * 2
            width = x_max - x_min
            height = y_max - y_min
        # draw a rectangle for the die
        fig.add_shape(
            type='rect',
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            line=dict(color='green', width=4, dash='dash')
        )
        # add text to the top of the die
        fig.add_trace(go.Scatter(
            x=[x_min + width / 2],
            y=[y_max + 200],
            text=[f"Die {int(row['die'])}"],
            mode='text',
            textposition='top center',
            textfont=dict(size=12, color='green')
        ))
        yield fig
    
    fig.update_layout(
        width=1200,
        height=1200,
        showlegend=False,
        xaxis=dict(title='X coordinate (micron)'),
        yaxis=dict(title='Y coordinate (micron)'),
        title='Positions of Wafer Blocks and Dies'
    )

    yield fig

    ss.progress_bar.progress(100, text='plotting completed')
    # update the layout of the figure


    # return fig


st.title('Wafer Structure Parser')



if 'wafer_structure_df' not in ss:
    ss.wafer_structure_df = pd.DataFrame()
if 'wafer_layout_fig' not in ss:
    ss.wafer_layout_fig = None
if 'row_count' not in ss:
    ss.row_count = 0
if 'column_count' not in ss:
    ss.column_count = 0
if 'same_row_col_counts' not in ss:
    ss.same_row_col_counts = False
if 'file_format' not in ss:
    ss.file_format = None
if 'file_name' not in ss:
    ss.file_name = None


#### File Upload
uploaded_file = st.file_uploader("upload a GDS or a CSV file", type=["GDS", "CSV"])

#### following code is to handle the uploaded file
if uploaded_file is not None:

    # check if the uploaded file is different from the previous uploaded file
    if uploaded_file.name != ss.file_name:
        ss.wafer_structure_df = pd.DataFrame()
        ss.wafer_layout_fig = None
        ss.file_name = uploaded_file.name if uploaded_file is not None else None
    
    # for GDS file
    if uploaded_file.name.lower().endswith('.gds'):
        ss.file_format = 'gds'
        # cannot read the file directly, need to save it to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gds") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        ## GSD parser options
        with st.expander("GDS parser options", expanded=True):
            st.write("### GDS parser configurations")
        ## 1st step: get and make user select the top cells
            top_cells_dict = get_top_level_cells(tmp_file_path)
            # user input for selecting the top cell
            selected_cell = st.selectbox("select top cells", list(top_cells_dict.keys()))
            # pad size 
            pad_length = st.number_input('pad length in micron', value=120)
            pad_width = st.number_input('pad width in micron', value=120)
            # probe spacing
            probe_spacing = st.number_input('probe spacing in micron', value=240, key='probe_spacing')
            # strict match option
            strict_match = st.toggle(
                label='strict match',
                help="""
                If strict match is disabled, the parser will match pads by the area of the pads, and vertices number (4 for rectangle pads) of the pads.
                If strict match is enabled, the parser will only consider pads that have the exact same length and width as the specified values.
                """
            )
            # group to die option
            group_to_die = st.toggle(
                label='group blocks to dies', 
                value=True,
                help="""
                If enabled, the parser will group blocks into dies based on the relative positions of the blocks. 
                Else, all blocks will be considered to be in the same die.
                """
            )

            if st.button('parse'):
                # progress bar
                ss.progress_bar = st.progress(0, text='parsing GDS file, please wait...')
        ## 2nd step: flatten the selected cell and find the pads that match the specified dimensions
                flattened_selected = flatten_selected(top_cells_dict, selected_cell)
                pad_coords = find_pads(flattened_selected, pad_length, pad_width, strict_match=strict_match)

                ss.progress_bar.progress(50, text='GDS file parsed successfully, processing data...')
                # if no pads found, stop the script and show a warning
                if not pad_coords:
                    st.warning("no qualified pads found in the selected cell")
                    st.stop()
                # if pads found, process the data
                else:
        ## 3rd step: parse the pad coordinates and classify the pads into blocks and dies (if group_to_die is enabled) and returned in dictionary format
            ## at this step the pad will be grouped into structures and keep only the top left coordinate
            ## pads are grouped according to the given probe layout, only 2*4 for now.
            ## but it can be extensively modified to support more probe layouts
                    wafer_structure = parse_pad_coords(pad_coords, group_to_die=group_to_die, pin_spacing=probe_spacing)
        ## 4th step: convert the wafer structure dictionary to a pandas dataframe
                    ss.wafer_structure_df = wafer_structure_to_dataframe(wafer_structure)
        ## 5th step: classify the blocks into rows and columns according to the relative positions of the blocks
                    classify_blocks_into_rows_and_columns(ss.wafer_structure_df, inplace=True)
                    # classify_rows_and_columns_in_blocks(ss.wafer_structure_df, inplace=True)
        ## 6th step: index the blocks in each die
                    index_blocks_in_die(ss.wafer_structure_df, inplace=True)
        ## 7th step: classify the rows and columns of the structures within each block
                    ss.wafer_structure_df, ss.same_row_col_counts, ss.row_count, ss.column_count = classify_rows_and_columns_in_blocks(ss.wafer_structure_df, inplace=True)
                    
                    ss.progress_bar.progress(100, text='data processed successfully, showing the data...')

    # for CSV file
    elif uploaded_file.name.lower().endswith('.csv'):
        ss.file_format = 'csv'
        if st.button('load CSV file', type='primary'):
            ss.wafer_structure_df = pd.read_csv(uploaded_file)

            if 'die' not in ss.wafer_structure_df.columns:
                st.warning("die column is missing, die index for each block will be assigned to 1")
                ss.wafer_structure_df['die'] = 1

            ## TODO: if really desiered, the need of 'block' column can be removed
            ##       by adding some parts of step 3 of GDS parser, but this would need to changed 
            ##       the source code of GDS parser functions to adapt to the new format
            if ('block' or 'x' or 'y') not in ss.wafer_structure_df.columns:
                st.warning("block, x or y column is missing, the uploaded file is not in the correct format")
                st.stop()

            ss.progress_bar = st.progress(0, text='CSV file loaded successfully, processing data...')
            try:
                ## replication of the last three steps of gds parser
                classify_blocks_into_rows_and_columns(ss.wafer_structure_df, inplace=True)
                # classify_rows_and_columns_in_blocks(ss.wafer_structure_df, inplace=True)
                index_blocks_in_die(ss.wafer_structure_df, inplace=True)
                ss.wafer_structure_df, ss.same_row_col_counts, ss.row_count, ss.column_count = classify_rows_and_columns_in_blocks(ss.wafer_structure_df, inplace=True)
            except Exception as e:
                st.warning(f"Error: {e}, please check the format of the uploaded file")
                ss.progress_bar.empty()
                st.stop()

            ss.progress_bar.progress(100, text='data processed successfully, showing the data...')

    else:
        st.warning("file format not supported")
        st.stop()

    ## local fragment, has to put it here
    @st.fragment
    # bounded as a fragment to avoid re-running the whole script when the button is clicked
    def wafer_df_process():
        if not ss.wafer_structure_df.empty:
            wafer_data_container = st.container()
            wafer_data_container_cols = wafer_data_container.columns([7,3])
                
            with wafer_data_container_cols[0]:
                # show the wafer structure dataframe
                st.write("### Wafer Structure Data")
                st.dataframe(ss.wafer_structure_df) 
         
            with wafer_data_container_cols[1]:
                # wafer structure summary
                st.write("### Wafer Structure Summary")
                st.write(f"Number of Dies: {ss.wafer_structure_df['die'].nunique()}")
                st.write(f"Number of Blocks: {ss.wafer_structure_df['block'].nunique()}")
                st.write(f"Number of Block Rows: {ss.wafer_structure_df['block_row'].nunique()}")
                st.write(f"Number of Block Columns: {ss.wafer_structure_df['block_column'].nunique()}")

                st.write(f"Same Row and Column Counts within all blocks: {ss.same_row_col_counts}")
                if not ss.same_row_col_counts:
                    st.warning(
                        """
                        Not all blocks have exactly the same row and column counts\n
                        i.e. the blocks are not in the same layout\n
                        information below about the DUT count is the maximum count of rows and columns in all blocks
                        """
                    )

                st.write(f"Number of DUT Rows in Blocks: {ss.row_count}")
                st.write(f"Number of DUT Columns in Blocks: {ss.column_count}")

                st.write(f'Number of DUT detected: {len(ss.wafer_structure_df)}')

            ### additional step: a different way of checking the structure layout
            same_layout_for_blocks = check_blocks_relative_positions(ss.wafer_structure_df)
            if not same_layout_for_blocks:
                st.warning("Not all blocks have exactly the same layout")
            
        ## Reindexing and labeling options
            with st.expander("Reindexing Options", expanded=False):
                indexing_container = st.container()
                indexing_container_cols = indexing_container.columns(3)
                # for absolute block and die indexing
                for i, level in enumerate(['die', 'block']):
                    with indexing_container_cols[i]:
                        st.write(f"### Reassign {level} index")
                        st.selectbox('primary axis', ['x', 'y'], index=1, key=f'{level}_primary')
                        st.toggle('x ascending', value=True, key=f'{level}_x_asc')
                        st.toggle('y ascending', value=False, key=f'{level}_y_asc')
                        if st.button(f'reassign {level} index'):
                            reassign(ss.wafer_structure_df, 
                                    level=level, 
                                    primary=ss[f'{level}_primary'], 
                                    x_asc=ss[f'{level}_x_asc'], 
                                    y_asc=ss[f'{level}_y_asc'], 
                                    inplace=True)
                # for block indexing in die
                with indexing_container_cols[2]:
                    st.write("### Reindex blocks in die")
                    st.selectbox('primary axis', ['x', 'y'], index=1, key='label_primary')
                    st.toggle('x ascending', value=True, key='label_x_asc')
                    st.toggle('y ascending', value=False, key='label_y_asc')
                    if st.button('label blocks in die'):
                        index_blocks_in_die(ss.wafer_structure_df, 
                                        primary=ss['label_primary'], 
                                        x_asc=ss['label_x_asc'], 
                                        y_asc=ss['label_y_asc'], 
                                        inplace=True)
                    
            with st.expander("Labeling Options", expanded=False):
                labeling_container = st.container()
                labeling_container_cols = labeling_container.columns(2)
                # for rows 
                with labeling_container_cols[0]:
                    st.write("### Create new labels for in block rows")
                    st.text_input('In block row annotation', key='row_annotation')
                    with st.container(border=True, height=300):
                        for i in range(1, ss.row_count + 1):
                            st.write(f"row {i}")
                            st.text_input(f"row {i} label", key=f'row_{i}_label')
                # for columns
                with labeling_container_cols[1]:
                    st.write("### Create new labels for in block columns")
                    st.text_input('In block column annotation', key='column_annotation')
                    with st.container(border=True, height=300):
                        for i in range(1, ss.column_count + 1):
                            st.write(f"column {i}")
                            st.text_input(f"column {i} label", key=f'column_{i}_label')
                with labeling_container:
                    if st.button('apply labels'):
                        check_passed = True
                        # check if all labels are filled
                        for i in range(1, ss.row_count + 1):
                            if ss[f'row_{i}_label'].strip() == '':
                                st.warning(f"row {i} label is empty")
                                check_passed = False
                                # st.stop()
                        for i in range(1, ss.column_count + 1):
                            if ss[f'column_{i}_label'].strip() == '':
                                st.warning(f"column {i} label is empty")
                                check_passed = False
                                # st.stop()
                        if ss['row_annotation'].strip() == '' or ss['column_annotation'].strip() == '':
                            st.warning("annotation is empty")
                            check_passed = False
                        
                        # check if row and column annotation are different
                        if ss['row_annotation'].strip() == ss['column_annotation'].strip():
                            st.warning("row and column annotation should be different")
                            check_passed = False
                            # st.stop()
                        # if all checks passed, apply the labels
                        if check_passed:
                            apply_labels_to_blocks(
                                ss.wafer_structure_df, 
                                [ss[f'row_{i}_label'] for i in range(1, ss.row_count + 1)],
                                ss['row_annotation'],
                                [ss[f'column_{i}_label'] for i in range(1, ss.column_count + 1)],
                                ss['column_annotation'],
                                inplace=True
                            )
                            # st.success("labels applied successfully")
                            st.rerun()
        ## save the wafer structure
            name_value = ss.wafer_structure_df['design'].values[0] if 'design' in ss.wafer_structure_df else ''
            design_name = st.text_input('Name this wafer design', value=name_value)
            if os.path.exists(os.path.join(coord_file_save_folder, f"{design_name}_wafer_structure.csv")):
                st.warning(f"design name {design_name} already exists, by saving this design, the existing design will be overwritten")
            save_file = st.button('Save this wafer design')
            # if st.button('Add this wafer design'):
            if save_file:
                if design_name.strip() == '':
                    st.warning("design name is empty, please enter a name")
                # if os.path.exists(os.path.join(coord_file_save_folder, f"{design_name}_wafer_structure.csv")):
                #     st.warning(f"design name {design_name} already exists, please choose another name")
                else:
                    # if os.path.exists(os.path.join(coord_file_save_folder, f"{design_name}_wafer_structure.csv")):
                    #     filename_doublecheck_window("design name already exists, do you want to overwrite the existing design?")
                    ss.wafer_structure_df['design'] = design_name
                    save_path = os.path.join(coord_file_save_folder, f"{design_name}_wafer_structure.csv")
                    ss.wafer_structure_df.to_csv(save_path, index=False)
                    st.success(f"wafer design saved at {save_path}")

        
        ## not real time plotting
            #     if st.button('plot the wafer'):
            #         if ss.file_format == 'csv':
            #             ss.wafer_layout_fig = plot_wafer(ss.wafer_structure_df)
            #         else:
            #             # ss.wafer_layout_fig = plot_wafer(ss.wafer_structure_df, probe_spacing)
            #             ss.wafer_layout_fig = plot_wafer(ss.wafer_structure_df, ss['probe_spacing'])

            # if ss.wafer_layout_fig is not None:
            #     st.info("""This plot is a visual representation of the wafer structure, 
            #             showing the relative positions of blocks and dies. 
            #             The depicted areas and distances between blocks and dies 
            #             are approximate and may not reflect exact measurements.""")
            #     st.plotly_chart(ss.wafer_layout_fig, use_container_width=False)


        ## realtime plotting that looks cool but probably not necessary
            with st.container(border=True):
                st.write("### Wafer Plot")
                plot_cols = st.columns([7,3])
                with plot_cols[0]:
                    start_plotting = st.button('Plot the Wafer', use_container_width=True, type='primary')
                    st.info("""
                            The plot only provides a visual representation of the wafer structure, 
                            showing the relative positions of blocks and dies. 
                            The depicted areas and distances between blocks and dies 
                            are approximate and may not reflect exact measurements.
                            Also, due to the large amount of data, the plot may take some time to generate, and the normal operation of the app may be affected.
                            """)
                    with st.container(height= 1200,border=True): # fix the height of the container so that the plot does not resize when generating the plot
                        image_placeholder = st.empty()  # for displaying the plot
                        if ss.wafer_layout_fig is not None:
                            image_placeholder.plotly_chart(ss.wafer_layout_fig, use_container_width=True, key='wafer_layout_fig')
                            # image_placeholder.plotly_chart(ss.wafer_layout_fig, use_container_width=False)
                        
                        if start_plotting:
                            ss.wafer_layout_fig = None
                            for fig in wafer_plot_generator(ss.wafer_structure_df, ss['probe_spacing'] if ss.file_format == 'gds' else 120):
                                with image_placeholder:
                                    st.plotly_chart(fig, use_container_width=True, clear_on_update=True)
                            ss.wafer_layout_fig = fig
                with plot_cols[1]:
                    st.empty()

    wafer_df_process()


