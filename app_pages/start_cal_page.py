import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import config as sys_config

import re

import json
import subprocess


st.title('Welcome to The Parametric Test System')




if 'prev_hwstu' not in st.session_state:
    st.session_state.prev_hwstu = {}

if 'running' not in st.session_state:
    st.session_state.running = False


check_dict = {
    "SMU": {
        "selftest": False,
        "acal": None,
    },
    "DMM": {
        "selftest": False,
        "acal": None,
    }
}

caltest_container = st.container(border=True)

with caltest_container:
    st.write('### Before any measurement, it is recommended to perform all of the following calibration and self-test tasks:')
    smu_col, dmm_col = st.columns(2)
    smu_col_container = smu_col.container(border=True)
    dmm_col_container = dmm_col.container(border=True)

with smu_col_container:
    ## checklist for SMU
    st.write('#### SMU Calibration/Self-Test')
    if st.checkbox('Calibrate SMU', help='Perform Self-Calibration on Source Measure Unit', value=True):
        check_dict['SMU']['acal'] = True
    else:
        check_dict['SMU']['acal'] = False

    if st.checkbox('Selftest SMU', help='Perform Self-testing on Source Measure Unit', value=True):
        check_dict['SMU']['selftest'] = True
    else:
        check_dict['SMU']['selftest'] = False

with dmm_col_container:
    ## checklist for DMM
    st.write('#### DMM Calibration/Self-Test')
    if st.checkbox('Calibrate DMM', help='Perform Self-Calibration on Digital Multimeter', value=True):
        st.selectbox(
            'DMM Calibration Type', 
            ["DCV", "ALL"], 
            index=0, 
            help="""
            DCV: DC Voltage Calibration, takes about 3 minutes to complete\n
            ALL: All Calibration, takes about 15 minutes to complete\n
            You will need to run the ALL calibration on restart of the DMM.
            For more information, please refer to the 3458A calibration manual.
            """, 
            key='dmm_acal'
        )
        check_dict['DMM']['acal'] = st.session_state.dmm_acal
    else:
        check_dict['DMM']['acal'] = None

    if st.checkbox('Selftest DMM', help='Perform Self-testing on Digital Multimeter', value=True):
        check_dict['DMM']['selftest'] = True
    else:
        check_dict['DMM']['selftest'] = False


### Task Control Panel
task_control_panel = st.container(border=True)
with task_control_panel:
    st.write("### Control & Status Panel")
# start_button_col, stop_button_col = task_control_panel.columns([1, 1])

# if st.button('Deploy task'):
if 'process' not in st.session_state:
    st.session_state.process = None
    st.session_state.running = False
    st.session_state.running_task_name = None

script_path = os.path.join(os.getcwd(), "utils/cal_utils.py")

with task_control_panel:
    start_button = st.button("Initiate Task", type='primary' if not st.session_state.running else 'secondary', use_container_width=True)

#### necessary variables for subprocess
if 'cal_subprogress' not in st.session_state:
    st.session_state.cal_subprogress = 0

if 'cal_subresult' not in st.session_state:
    st.session_state.cal_subresult = None

if 'cal_newest_status' not in st.session_state:
    st.session_state.cal_newest_status = None

if 'cal_tested' not in st.session_state:
    st.session_state.cal_tested = 0

if 'cal_tobe_tested' not in st.session_state:
    st.session_state.cal_tobe_tested = 0

if 'cal_subpro_fnl_status' not in st.session_state:
    st.session_state.cal_subpro_fnl_status = None

if 'subprocess_stop_flag' not in st.session_state:
    st.session_state.subprocess_stop_flag = False

if 'measurement_event' not in st.session_state:
    st.session_state.measurement_event = ''

if 'running_task_name' not in st.session_state:
    st.session_state.running_task_name = None

# run subprocess if there's no task running
if start_button and not st.session_state.running:


    # initialize the temp directory
    if not os.path.exists('temp'):
        os.makedirs('temp')

    check_dict_path = 'temp/check_dict.json'
    with open(check_dict_path, 'w') as f:
        json.dump(check_dict, f)


    st.session_state.process = subprocess.Popen(
        [
            "python", script_path,
            "--check_dict", check_dict_path,
        ],
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        encoding='utf-8' # must be set to utf-8 to avoid decoding error
    )


    st.session_state.running = True
    # st.session_state.running_task_name = 'Instrument_Self-Calibration/Self-Test'
    st.session_state.running_task_name = sys_config.SELF_CAL_TEST_NAME
    st.session_state.cal_tobe_tested = 0
    st.session_state.cal_tested = 0
    st.rerun()

elif start_button and st.session_state.running:
    if st.session_state.running_task_name is not None:
        task_control_panel.info(f"Task: {st.session_state.running_task_name} is currently running, please wait...")
    else:
        task_control_panel.info("Task is already running, please wait...")


# update the status of the subprocess
@st.fragment(run_every='1s')
def status_part():
    with st.container(border=True):
        st.write("#### Real-Time Task Status")

        st.button("Refresh")
        log_placeholder = st.empty()  # placeholder for log output
        if not st.session_state.running:
            st.session_state.cal_subprogress = 0
            # st.session_state.running_task_name = None

        if 'subprocess_progress_bar' not in st.session_state:
            st.session_state.subprocess_progress_bar = st.progress(0)

        # progress_bar_placeholder = st.empty()  # placeholder for progress bar
        result_status_placeholder = st.empty()  # placeholder for result status


        log_placeholder.write(st.session_state.cal_newest_status)
        # st.session_state.subprocess_progress_bar.progress(st.session_state.cal_subprogress/100)
        st.progress(st.session_state.cal_subprogress/100)
        result_status_placeholder.write(st.session_state.cal_subresult)
        
    if st.session_state.running_task_name == sys_config.SELF_CAL_TEST_NAME or st.session_state.running_task_name == None:
        if st.session_state.running:
            st.session_state.cal_subresult = f"Task Status: Task running..."

        # if st.session_state.running and st.session_state.process is not None:
        if st.session_state.running and st.session_state.process is not None:
            process = st.session_state.process
            # for line in process.stdout:
            if process.poll() is None:   
                # print([line for line in process.stdout])
                line = process.stdout.readline()
                
                # real-time output
                # log_placeholder.text(line.strip())
                st.session_state.cal_newest_status = line.strip()
                match = re.search(r"(\d+) out of (\d+) tasks performed", line)
                if match:
                    st.session_state.cal_tested = int(match.group(1))       # number of tasks performed
                    st.session_state.cal_tobe_tested = int(match.group(2))        # total number of tasks to be performed
                    st.session_state.cal_subprogress = int(st.session_state.cal_tested / st.session_state.cal_tobe_tested * 100)  # progress percentage
                    # print(st.session_state.cal_subprogress)

            # check if the process has completed
            if process.poll() is not None:  # if the poll() method returns a value, the process has completed

                if process.returncode != 0: # problem occurred
                    # result_status_placeholder.error(f"Command failed with return code {process.returncode}")
                    st.session_state.cal_subresult = f"Task Status: Task failed with return code {process.returncode}, log:{[line.strip() for line in process.stdout]}"
                    st.session_state.cal_subpro_fnl_status = 'failed'
                    # print(f"Task failed with return code {process.returncode}, log:{[line.strip() for line in process.stdout]}")
                else: # completed successfully
                    result_status_placeholder.success("Task completed successfully")
                    st.session_state.cal_subresult = "Task Status: Task completed successfully"
                    st.session_state.cal_subpro_fnl_status = 'success' 
                    # print('Command completed successfully')

                # read all lines
                line = [line.strip() for line in process.stdout]
                line = line[-1] if line else ''
                
                
                st.session_state.cal_newest_status = line.strip()

                # reset the process variable
                st.session_state.running = False
                st.session_state.process = None
                st.session_state.running_task_name = None
                st.rerun()        
with task_control_panel:
    status_part()


## Fetch the status of the most recent task
status_file = sys_config.CAL_STATUS_FILE_PATH
with st.container(border=True):
    st.write('### Most Recent Task')

    if os.path.exists(status_file):

        try:
            with open(status_file, 'r', encoding='utf-8') as file:
                stu = json.load(file)
            exception = stu.get('exception', None)
            if exception:
                st.warning(f"Exception occurred during the task: {exception}")
            status_msg = stu.get('status', None)
            st.write('Result from recent task:', status_msg)
            time = stu.get('time', None)
            st.write('Time:', time)
            tasks = stu.get('tasks', None)
            smu_tasks = tasks.get('SMU', None)
            dmm_tasks = tasks.get('DMM', None)
            smu_stest = smu_tasks.get('selftest', None)
            smu_acal = smu_tasks.get('acal', None)
            dmm_stest = dmm_tasks.get('selftest', None)
            dmm_acal = dmm_tasks.get('acal', None)

            task_performed = ['SMU_self-test' if smu_stest else None, 'SMU_self-calibration' if smu_acal else None, 'DMM_self-test' if dmm_stest else None, f'DMM_self-calibration:{dmm_acal}' if dmm_acal else None]
            st.write('Tasks performed:', [task for task in task_performed if task is not None])

        except Exception as e:
            st.warning(f'Error: {e}')



