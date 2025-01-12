
PB_GPIB_ADDR = 1
SMU_GPIB_ADDR = 23
SWM_GPIB_ADDR = 22
DMM_GPIB_ADDR = 12



STABILITY_TEST_NAME = 'Stability_Test'
SELF_CAL_TEST_NAME = 'Instrument_Self-Calibration/Self-Test'
WAFER_CHARA_TEST_NAME = 'Wafer-Level_Characterisation'



SMU_CHANNEL_MAP = { # SMU channel configuration, obtained from testing 
    'smu1': 1,
    'smu2': 3,
    'smu3': 4,
    'smu4': 5,
    'vs1': 6,
    'vs2': 6
}


SWM_PIN_ALIAS_LIST = ["u1", "u2", "u3", "u4", "d1", "d2", "d3", "d4"]
SWM_CONNECTED_RESOURCES_LIST = [
    "floating",
    "smu1 (40uV-100V 100mA)",
    "smu2 (40uV-200V 1A)",
    "smu3 (40uV-100V 100mA)",
    "smu4 (40uV-100V 100mA)",
    "vs1 (+/- 40V)",
    "vs2 (+/- 40V)",
    "gnd (SMU ground)",
    "dmm_hi",
    "dmm_lo"
    
]

SWM_CONNECTED_RESOURCES_ALIAS_DICT = {
    "smu1 (40uV-100V 100mA)": "smu1",
    "smu2 (40uV-200V 1A)": "smu2",
    "smu3 (40uV-100V 100mA)": "smu3",
    "smu4 (40uV-100V 100mA)": "smu4",
    "vs1 (+/- 40V)": "vs1",
    "vs2 (+/- 40V)": "vs2",
    "gnd (SMU ground)": "gnd",
    "dmm_hi": "dmm_hi",
    "dmm_lo": "dmm_lo",
    "floating": None
}


SWM_ALIAS_DICT = {
    'pins': {'u1':17,'u2':14,'d2':23,'d1':20,'d4':5,'u4':8,'d3':2,'u3':11},
    'ports': {'smu1':1,'smu2':2,'smu3':3,'smu4':4,'vs1':5,'vs2':6,'gnd':7,'dmm_hi':8,'dmm_lo':9}
}


from pathlib import Path

# Paths for measurement configuration files
PIN_MAP_FILE_PATH = Path('measurement_cfgs') / 'pin_maps.json'
CURR_CFG_FILE_PATH = Path('measurement_cfgs') / 'curr_cfg.json'
CFG_MAPS_FILE_PATH = Path('measurement_cfgs') / 'cfg_maps.json'
ALIAS_LW_MAP_FILE_PATH = Path('measurement_cfgs') / 'alias_lw_map.json'

# Temporary paths
TEMP_FOLDER_PATH = Path('temp')
TEMP_FILTERED_DF_PATH = TEMP_FOLDER_PATH / 'filtered_df.pkl'
TEMP_PIN_CFGS_PATH = TEMP_FOLDER_PATH / 'pin_cfgs.json'
TEMP_CFG_MAP_PATH = TEMP_FOLDER_PATH / 'cfg_map.json'
TEMP_CURR_CFGS_PATH = TEMP_FOLDER_PATH / 'curr_cfgs.json'
TEMP_SMU_CHANNEL_MAP_PATH = TEMP_FOLDER_PATH / 'smu_channel_map.json'
TEMP_STOP_FLAG_PATH = TEMP_FOLDER_PATH / 'stop_flag.txt'

# Status files
STATUS_FILE_PATH = TEMP_FOLDER_PATH / 'stu.json'
CAL_STATUS_FILE_PATH = TEMP_FOLDER_PATH / 'cal_stu.json'
STABILITY_STATUS_FILE_PATH = TEMP_FOLDER_PATH / 'stb_stu.json'

# Data collection folders
STABILITY_MEASUREMENT_FOLDER = Path('data_collection/measurements/stability_results')
STABILITY_PROCESSED_DATA_FOLDER = Path('data_collection/processed_data/stability_tests')

RAW_MEASUREMENT_OUTPUT_FOLDER = Path('data_collection/measurements/raw')

WAFER_STRUCTURE_FOLDER_PATH = Path('data_collection/wafer_structure_files')

GROUPED_MEASUREMENTS_FOLDER = Path('data_collection/measurements/grouped_results')


VARIATION_OBJECT_IGNO_LIST = ['wafer', 'design', 'x', 'y', 'block', 'die', 'block_row', 'block_column']

PIN_VARIANTS_TOO_LONG_MSG = """
                                The selected pin variation has too many unique items, 
                                this might cause problem in display or need a long time to re-render the page, 
                                do you wish to continue?
                            """
CURR_VARIANTS_TOO_LONG_MSG = """
                                The selected current variation has too many unique items, 
                                this might cause problem in display or need a long time to re-render the page, 
                                do you wish to continue?
                            """