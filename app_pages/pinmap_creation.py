

import streamlit as st
import numpy as np
import pandas as pd
import json
import config as sys_config

def save_to_json(filename, content):
    with open(filename, 'w') as json_file:
        json.dump(content, json_file, indent=4)


# st.set_page_config(layout="wide")
st.title("Pin Map Configuration")

# call this for formatting only
st.markdown("""
    <style>
    .selectbox-title {
        text-align: center;
        display: block;
        margin-bottom: 8px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Load the system configuration
pin_map_file = sys_config.PIN_MAP_FILE_PATH
pin_keys = sys_config.SWM_PIN_ALIAS_LIST

resources = sys_config.SWM_CONNECTED_RESOURCES_LIST



resource_dict = sys_config.SWM_CONNECTED_RESOURCES_ALIAS_DICT


# Initialize session state to store the selected resources
if "selected_resources" not in st.session_state:
    st.session_state.selected_resources = ["floating"] * len(pin_keys)
if "reassign" not in st.session_state:
    st.session_state.reassign = False
if "pin_cfgs" not in st.session_state:
    with open(pin_map_file, 'r') as file:
        exist_maps = json.load(file)
        st.session_state.pin_cfgs = exist_maps

# Function to check and reassign resources
def reassign_resource(selected_resource, index):
    for i, resource in enumerate(st.session_state.selected_resources):
        if resource == selected_resource and i != index:
            st.session_state.selected_resources[i] = "floating"
    st.rerun()

@st.fragment
def pin_assignment():
    # Create the container for the pin assignment
    connection_design_container = st.container(border=True)
    connection_design_container.markdown("<h2 class='selectbox-title'>Create New Pin Map</h2>", unsafe_allow_html=True)

    
    # NOTE: when modifying the number of rows, or columns, pin_keys should be updated accordingly to match the number of widgets in each row
    col_num = 4
    row_num = 2

    # Split the container into two rows
    row1 = connection_design_container.columns(col_num)
    html_container = connection_design_container.container()
    if row_num == 2:
        row2 = connection_design_container.columns(col_num)
    else:
        row2 = []

    # Loop through both rows and assign widgets with keys
    for i, col in enumerate(row1 + row2):
        # for display between the two rows
        if i == col_num:
            data = st.session_state.selected_resources
            # html_container.markdown("<hr>", unsafe_allow_html=True)
            for row in range(0, len(data), col_num):
                cols = html_container.columns(col_num)  # Create 4 columns to have 4 blocks in each row
                
                for x in range(col_num):
                    if row + x < len(data):
                        value = data[row + x]
                        # Set background color to red if value is "Yes"
                        color = "red" if value != "floating" else "#FFFFFF"
                        # color = "blue" if value in ("dmm_lo", "dmm_hi") else color
                        color = '#5581FA' if value == "dmm_lo" else color
                        color = '#DB7C54' if value == "dmm_hi" else color
                        
                        color = "green" if value == "gnd (SMU ground)" else color
                        
                        # Render the square block
                        with cols[x]:
                            st.markdown(
                                f"""
                                <div style="
                                    height: 100px;
                                    width: 100px;
                                    background-color: {color};
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    border: 1px solid black;
                                    margin-bottom: 10px;
                                    margin-left: auto;
                                    margin-right: auto;
                                ">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        tile = col.container(border=True)
        tile.markdown(f"<p style='text-align: center; font-weight: bold;'>{pin_keys[i].capitalize()}</p>", unsafe_allow_html=True)
        if st.session_state.selected_resources[i] == "floating":
            selected_index = 0
        else:
            try:
                selected_index = resources.index(st.session_state.selected_resources[i])
                # selected_index = resources.index(st.session_state.selected_resources[i]) + 1
            except ValueError:
                selected_index = 0
        # print(selected_index)
        selected_resource = tile.selectbox(
            label=pin_keys[i].capitalize(),
            # options=[None] + resources,
            options=resources,
            index=selected_index,
            key=pin_keys[i],
            label_visibility="hidden"
        )
        
        # Update the session state with the selected resource
        previous_resource = st.session_state.selected_resources[i]
        st.session_state.selected_resources[i] = selected_resource
        # Reassign the resource if it's already selected by another box
        if selected_resource != previous_resource:
            st.session_state.reassign = reassign_resource(selected_resource, i)
    
    with connection_design_container:
        # Display a warning if more than one SMU channel is selected
        count = sum(1 for s in st.session_state.selected_resources if (s.startswith("smu") or s.startswith("vs")))
        if count > 1:
            st.warning("more than one SMU channel is selected, this function is not supported yet, please select only one SMU channel")
        elif count == 0:
            st.warning("no SMU channel is selected, you need to select one SMU channel to force the current")


        quick_func_buttons_container = st.container(border=True) # Create a container for the quick functions
        with quick_func_buttons_container:
            # NOTE: the first three quick functions adapt to the current 2*4 layout only, the last one is a general function
            
            st.write("### Quick Functions") # Display the title of the quick functions
            quick_funcs_container = st.container()
            
            quick_func_col1, quick_func_col2, quick_func_col3, quick_func_col4 = quick_funcs_container.columns(4)
            # clockwise rotation
            if quick_func_col1.button(
                "Rotate Resources Clockwise", 
                use_container_width=True,
                help="This button will rotate the selected resources clockwise (excluding 'floating')"
            ):
                # Reorder the list to [a, b, c, d, h, g, f, e]
                reordered_resources = st.session_state.selected_resources[:4] + st.session_state.selected_resources[-1:-5:-1]
                
                # Rotate the selected items clockwise, excluding "floating"
                selected_only = [res for res in reordered_resources if res != "floating"]

                # if len(selected_only) <= 8:
                # if len(selected_only) == 4:
                rotated = selected_only[-1:] + selected_only[:-1]
                idx = 0
                for i in range(len(reordered_resources)):
                    if reordered_resources[i] != "floating":
                        reordered_resources[i] = rotated[idx]
                        idx += 1
                # Map list back to original order
                st.session_state.selected_resources = reordered_resources[:4] + reordered_resources[-1:-5:-1]
                st.rerun()

            # mirror across Y-axis
            if quick_func_col2.button(
                "Mirror Resources", 
                use_container_width=True,
                help="This button will mirror the selected resources within the same row (excluding 'floating')"
            ):
                    # Split the list into two rows
                    half = len(st.session_state.selected_resources) // 2
                    first_row = st.session_state.selected_resources[:half]
                    second_row = st.session_state.selected_resources[half:]
                    
                    # Mirror each row excluding "floating"
                    mirrored_first_row = [res for res in first_row if res != "floating"][::-1]
                    mirrored_second_row = [res for res in second_row if res != "floating"][::-1]
                    
                    idx = 0
                    for i in range(len(first_row)):
                        if first_row[i] != "floating":
                            first_row[i] = mirrored_first_row[idx]
                            idx += 1
                    
                    idx = 0
                    for i in range(len(second_row)):
                        if second_row[i] != "floating":
                            second_row[i] = mirrored_second_row[idx]
                            idx += 1
                    
                    # Combine the rows back into the original list
                    st.session_state.selected_resources = first_row + second_row
                    
                    st.rerun()

            # mirror across X-axis
            if quick_func_col3.button(
                "Swap Rows", 
                use_container_width=True,
                help="This button will swap the two rows of selected resources"
            ):
                # Split the list into two rows
                half = len(st.session_state.selected_resources) // 2
                first_row = st.session_state.selected_resources[:half]
                second_row = st.session_state.selected_resources[half:]
                
                # Swap rows to mirror across X-axis
                st.session_state.selected_resources = second_row + first_row
                
                st.rerun()

            # clear all
            if quick_func_col4.button(
                "Clear All", 
                use_container_width=True,
                help="This button will clear all selected resources, set all resources to 'floating'"
            ):
                # Set all resources to "floating"
                st.session_state.selected_resources = ["floating"] * len(pin_keys)
                
                st.rerun()

        # save current configuration
        name = st.text_input("Configuration Name", placeholder="Specify the configuration polarity by using the format: ConfigName_P or ConfigName_N", help="ConfigName**_P** for positive polarity, ConfigName**_N** for negative polarity, e.g. LW300_P, LW300_N. !!! Without the postfix **_P** or **_N**, the configuration will treat as a general positive configuration")
        if st.button("save", type="primary"):
            if name is not None and name != "":
                resource_aliases = [resource_dict[res] for res in st.session_state.selected_resources]
                pin_port_config = dict(zip(pin_keys, resource_aliases))
                st.session_state.pin_cfgs[name] = pin_port_config
                save_to_json(pin_map_file, st.session_state.pin_cfgs)
                st.success("Configuration saved successfully")
                # # reload pin_cfgs
                # with open(pin_map_file, 'r') as file:
                #     exist_maps = json.load(file)
                #     st.session_state.pin_cfgs = exist_maps
                # st.rerun()

            else:
                st.warning("Please enter a name for the pin configuration")
                # st.stop()

pin_assignment()

#     ## Debugging purposes
#     # st.write("Current resource assignments:")
#     # for i, resource in enumerate(st.session_state.selected_resources):
#     #     st.write(f"{pin_keys[i].capitalize()}: {resource}")


@st.fragment()
def show_del_existing_maps():
    # Load the existing configurations
    df = pd.DataFrame(st.session_state.pin_cfgs)
    # display existing configurations
    save_del_container = st.container(border=True)
    save_del_container.markdown("<h2 class='selectbox-title'>Existing Pin Map Configurations</h2>", unsafe_allow_html=True)
    col_avaliable_cfg, col_select_del = save_del_container.columns([3, 1])
    col_avaliable_cfg.dataframe(df)
    col_avaliable_cfg.button("Refresh")
    to_del = col_select_del.selectbox(
        label="Select a configuration to detele", 
        options=list(st.session_state.pin_cfgs.keys()), 
        index=None, 
        placeholder="Select a configuration to delete"
    )
    # delete the selected configuration
    if col_select_del.button("Delete"):
        if to_del in st.session_state.pin_cfgs:
            del st.session_state.pin_cfgs[to_del]
            save_to_json(pin_map_file, st.session_state.pin_cfgs)
            st.rerun()
        elif to_del is None:
            col_select_del.warning("Please select a configuration to delete")
            st.stop()

show_del_existing_maps()
######################################
######################################

