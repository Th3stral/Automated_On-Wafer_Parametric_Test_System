import streamlit as st
import config as sys_config

from utils.cal_utils import hw_stu_check

st.set_page_config(page_title='Parametric Testing App', layout='wide')


if 'running' not in st.session_state:
    st.session_state.running = False
if 'prev_hwstu' not in st.session_state:
    st.session_state.prev_hwstu = {}

# Define the pages
pages = {
    "Prepreation": [
        st.Page("app_pages/start_cal_page.py", title="Hardware Calibration"),
        st.Page("app_pages/wafer_design_page.py", title="Import / Modify Wafer Design"),
        st.Page("app_pages/pinmap_creation.py", title="Pinmap Configuration"),
    ],
    "Test Single Structure":[st.Page("app_pages/stability_test.py", title="Stability Test"),],
    "Wafer-Map": [
        st.Page("app_pages/issue_mapping_event.py", title="Issue New Wafer Mapping Event"),
        st.Page("app_pages/inspect_wafermap.py", title="Wafer Map Report"),
    ],
}

# Sidebar util that checks the hardware status
def hw_stu_check_module():
    pb_addr = sys_config.PB_GPIB_ADDR   
    smu_addr = sys_config.SMU_GPIB_ADDR
    swm_addr = sys_config.SWM_GPIB_ADDR
    dmm_addr = sys_config.DMM_GPIB_ADDR
    with st.sidebar:
        ## TODO: considering to move this to subprocess, so the multi-click issue may be resolved
        # st.write('Press the button below to check the hardware status')
        st.info('Please DO NOT click the button multiple times at once, it may cause the probebench remote control system to crash and then you will need to reboot the probebench')

        check_button = st.button(
            'Check Hardware Status', 
            disabled=st.session_state.running
        )

        if check_button:
            if not st.session_state.running: # double check if there is a task running
                st.warning('Checking hardware status...\n Please **DO NOT** click the button again, multiple clicks may cause the probebench remote control system to crash')
                # st.session_state.running = True
                stu = hw_stu_check(
                    pb_addr=pb_addr, 
                    smu_addr=smu_addr, 
                    swm_addr=swm_addr, 
                    dmm_addr=dmm_addr
                )
                
            else:
                stu = st.session_state.prev_hwstu
                st.info('There is a hardware task running, please wait...')
            st.session_state.prev_hwstu = stu
            st.rerun()
        if st.session_state.prev_hwstu:
            st.write(st.session_state.prev_hwstu)


hw_stu_check_module()

pg = st.navigation(pages)
pg.run()