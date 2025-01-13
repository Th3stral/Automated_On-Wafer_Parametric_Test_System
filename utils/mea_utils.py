import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import csv
import time
from datetime import datetime


import pandas as pd
import pyvisa
import numpy as np


import argparse
import json
import pickle
import pandas as pd


import utils.log_utils as log_utils
from utils.hw_utils import SMUinterface, DMMinterface, PBinterface, SWMinterface, HardwareError


def hw_init(pb_addr=1, smu_addr=23, swm_addr=22, dmm_addr=12):
    # initialize the hardware
    rm = pyvisa.ResourceManager() # create the resource manager osbject

    # instantiate the hardware interfaces
    pb = PBinterface(rm, addr=pb_addr)
    pb.delay_time = 0.1

    dmm = DMMinterface(rm, addr=dmm_addr)
    smu = SMUinterface(rm, addr=smu_addr)
    matrix = SWMinterface(rm, addr=swm_addr)

    # disable the display of the DMM
    dmm._set_display('OFF')

    # set the aliases for the switch matrix
    alias_dict = sys_config.SWM_ALIAS_DICT

    matrix.set_aliases(alias_dict)
    return rm, pb, dmm, smu, matrix



def handle_exception(e, hw_module, raise_exception=True, additional_info=None):
    """
    Handles exceptions by categorizing and logging error messages.

    Args:
        e (Exception): The exception object raised.
        hw_module (str): Name of the hardware module where the exception occurred.
        raise_exception (bool, optional): Whether to re-raise the exception. Defaults to True.
        additional_info (str, optional): Additional context for the error message. Defaults to None.

    Returns:
        Exception or None: Returns the exception if `raise_exception` is True; otherwise, None.
    """
    add_info = additional_info if additional_info is not None else ''
    if isinstance(e, HardwareError):
        status_msg = f"{add_info} Error raised when executing hardware module: {hw_module}, when operating with {e.module}. Error: {e}"
        log_utils.log_status(hw_module, status_msg, level='ERROR')
        # on hardware error, communication still works, 
    elif isinstance(e, pyvisa.errors.VisaIOError):
        status_msg = f"{add_info} Error in communicating with the hardware when executing hardware module: {hw_module}. Error: {e}"
        log_utils.log_status(hw_module, status_msg, level='ERROR')
    elif isinstance(e, (ValueError, TypeError, KeyError, IndexError)):
        status_msg = f"{add_info} Error raised when processing values in software. Error: {e}"
        log_utils.log_status('software', status_msg, level='ERROR')
    else:
        status_msg = f"{add_info} Undefinded error occured when executing hardware module: {hw_module}. Error: {e}"
        log_utils.log_status(hw_module, status_msg, level='ERROR')

    if raise_exception:
        # raise e  # raise the exception to the calling function
        return e
    else:
        return None

def setup_measurement_record(design_name, pin_variation, curr_variation, output_folder):
    """
    Sets up a CSV file to record measurement data for a specific test or experiment.

    Args:
        design_name (str): The name of the design under test.
        pin_variation (str): Variation of pin configurations being measured.
        curr_variation (str): Variation of current settings being applied.
        output_folder (str): Folder path where the output file will be saved.

    Returns:
        tuple: 
            - measurements (file object): Open file object for writing measurements.
            - writer (csv.writer): CSV writer object for recording data.
            - event_name (str): Unique event name based on the design name and timestamp.
    """
    event_name = f"{design_name}_{datetime.now().strftime('%Y_%m_%d_%H%M')}"
    output_file = f"{output_folder}/{event_name}_raw.csv"
    output_open_mode = "w"  # Choose "a" for appending instead of overwriting

    measurements = open(output_file, mode=output_open_mode, newline="")
    writer = csv.writer(measurements)
    writer.writerow(["die", "block", pin_variation, curr_variation, "timestamp", "config", "force_I_setting","v_measure", "i_measure", "smu_status"])  # Header row
    output_file = f"{design_name}_{datetime.now().strftime('%Y_%m_%d_%H%M')}_raw.csv"
    return measurements, writer, event_name


def get_channels(pin_cfg, smu_channel_map):
    """
    Extracts the channel values from the SMU channel map based on the pin configuration.

    Args:
        pin_cfg (dict): Dictionary containing pin configurations with keys as pin names
                        and values as hardware identifiers (e.g., "smu1", "vs2").
        smu_channel_map (dict): Dictionary mapping SMU/VS identifiers to channel values.

    Returns:
        list: List of channel values corresponding to valid entries in the SMU channel map.

    Example:
        pin_cfg = {"pin1": "smu1", "pin2": "vs2", "pin3": None}
        smu_channel_map = {"smu1": 1, "vs2": 2}
        channels = get_channels(pin_cfg, smu_channel_map)
        # channels = [1, 2]
    NOTE: although this function support multiple channels, the current implementation only supports one channel
    """
    channel_values = []
    for key, value in pin_cfg.items():
        if value and (value.startswith("smu") or value.startswith("vs")):
            if value in smu_channel_map:
                channel_values.append(smu_channel_map[value])
    return channel_values


def forceI_measureV(smu, dmm, smu_channel, force_current, nrdgs, nplc, structure_cfg_info={}, writer = None, volt_comp=1):
    """
    Forces a constant current using the SMU and measures the resulting voltage with the DMM.

    Args:
        smu: SMU interface object.
        dmm: DMM interface object.
        smu_channel (int): SMU channel to use.
        force_current (float): Current to force using the SMU.
        nrdgs (int): Number of readings to take.
        nplc (float): DMM integration time in power line cycles.
        structure_cfg_info (dict): Configuration details of the structure under test.
        writer (csv.writer, optional): CSV writer object to record measurements. Defaults to None.
        volt_comp (float, optional): Voltage compliance for the SMU. Defaults to 1.

    Returns:
        tuple: (mean voltage, mean current, compliance flag, all voltage measurements)
    """

    global hw_module # for logging

    die = structure_cfg_info.get('die')  # Returns None if 'die' key is not present
    block = structure_cfg_info.get('block')  # Returns None if 'block' key is not present
    pin_variation = structure_cfg_info.get('pin_variation')
    curr_variation = structure_cfg_info.get('curr_variation')
    config_name = structure_cfg_info.get('pin_config')

    volt_mea = []
    curr_mea = []

    smu_comped = False

    hw_module = 'SMU'

    # turn on the channel
    smu._channel_on(smu_channel)
    log_utils.log_status(hw_module, f"SMU channel {smu_channel} turned ON", level='INFO')

    # zero the output
    smu._zero_output(smu_channel)
    log_utils.log_status(hw_module, f"SMU channel {smu_channel} output zeroed", level='INFO')

    # set the force current
    smu._set_force_current(smu_channel, 17, force_current, volt_comp) # set SMU constant current
    log_utils.log_status(hw_module, f"SMU channel {smu_channel} set to force current: {force_current}", level='INFO')

    time.sleep(0.2) # allow to settle

    # check for errors
    smu.check_errors()


    # start the measurement
    for i in range(nrdgs):
        # measure the voltage
        hw_module = 'DMM'
        v = dmm.dcv_measurement(nrdgs=1, nplc=nplc)
        v_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_utils.log_status(module_name=hw_module, status_message=f"Voltage measured: {v[0]}", level='INFO')

        # measure the current
        hw_module = 'SMU'
        current_value, smu_status, _, mea_type = smu.get_spot_current(smu_channel)
        log_utils.log_status(hw_module, f"SMU channel {smu_channel} measured {mea_type}: {current_value}, with status: {smu_status}", level='INFO')

        # write the measurements to the file
        if writer != None:
            writer.writerow([die, block, pin_variation, curr_variation, v_time, config_name, format(force_current, ".9f"), v[0], current_value, smu_status])

        # append the measurements to the lists
        volt_mea.extend(v)
        curr_mea.append(current_value)
        # print(f"voltage measurements: {volt_mea}\n mean voltage: {np.mean(volt_mea)}")
        if smu_status == 'C': # check if the SMU is in compliance
            smu_comped = True
    
    # zero the output
    smu._zero_output(smu_channel)
    log_utils.log_status(hw_module, f"SMU channel {smu_channel} output zeroed", level='INFO')
    # turn off the channel
    smu._channel_off(smu_channel)
    log_utils.log_status(hw_module, f"SMU channel {smu_channel} turned OFF", level='INFO')

    # check for errors
    smu.check_errors()

    hw_module = 'DMM'
    dmm.check_errors()

    # calculate the mean values
    meanV = np.mean(volt_mea)
    meanI = np.mean(curr_mea)

    return meanV, meanI, smu_comped, volt_mea


# raw_measurement_output_folder = 'data_collection/measurements/raw'
# def main_measurement(pb, dmm, smu, matrix, filtered_df, pin_cfgs, cfg_map, curr_cfgs, curr_variation, pin_variation, smu_channel_map, nrdgs=1, nplc=5, repeat=1, manual_mode=False):
def main_measurement(pb, dmm, smu, matrix, filtered_df, pin_cfgs, cfg_map, curr_cfgs, smu_channel_map, repeat=1, manual_mode=False, stb_flag=False, raw_measurement_output_folder = 'data_collection/measurements/raw'):
    """
    Executes measurements for structures in a filtered dataset using hardware interfaces.

    Args:
        pb, dmm, smu, matrix: Hardware interface objects.
        filtered_df (DataFrame): Filtered DataFrame containing structure details.
        pin_cfgs, cfg_map, curr_cfgs, smu_channel_map (dict): Hardware and test configurations.
        repeat (int): Number of measurement repetitions. Defaults to 1.
        manual_mode (bool): Skip automated chuck movement if True. Defaults to False.
        stb_flag (bool): Enable stability measurement logging. Defaults to False.
        raw_measurement_output_folder (str): Folder for raw measurement data. Defaults to 'data_collection/measurements/raw'.

    Returns:
        tuple: (results DataFrame, event name, exception object if any)
    """

    global hw_module # for logging

    exception_occurred = None  # for exception handling
    log_utils.configure_logger(log_filename='hardware_operations.log', level='INFO') # configure the logger

    # Validate input data
    if filtered_df.empty:
        raise ValueError("The filtered DataFrame is empty, cannot proceed with the measurement")
    
    # Extract configuration details from the DataFrame
    if '__curr_variation_object' in filtered_df.columns:
        curr_variation = filtered_df['__curr_variation_object'].iloc[0]
    elif 'material' in filtered_df.columns:
        curr_variation = 'material'
    else:
        raise ValueError("The current variation object is not found in the DataFrame")
    
    if '__pin_variation_object' in filtered_df.columns:
        pin_variation = filtered_df['__pin_variation_object'].iloc[0]
    elif 'structure' in filtered_df.columns:
        pin_variation = 'structure'
    else:
        raise ValueError("The pin variation object is not found in the DataFrame")
    
    if 'design' in filtered_df.columns:
        design_name = filtered_df['design'].iloc[0]
    else:
        design_name = 'unknown'
    if 'wafer_idx' in filtered_df.columns:
        wafer_idx = filtered_df['wafer_idx'].iloc[0]
    else:
        wafer_idx = 'unknown'

    # Initialize measurement record
    measurements, writer, event_name = setup_measurement_record(design_name, pin_variation, curr_variation, raw_measurement_output_folder)
    
    results_df = pd.DataFrame({})  # create an empty DataFrame to store the results
    # results_df = pd.DataFrame(columns=['die', 'block', pin_variation, curr_variation, 
    #                                    'alias', 'pin_config', 'pol', 'force_current', 
    #                                    'meanV', 'meanI', 'hit_comp', 'block_row', 'block_column'])

    raise_error = True

    structure_counter = 0
    event_counter = 0

    total_structures = len(filtered_df)


    try:
        # probe initialization
        print("Test started...")
        hw_module = 'PB'
        pbresp = pb._set_chuck_vacuum(vacuum_on=True) # turn on the vacuum
        log_utils.log_status(hw_module, f"Vacuum ON: {pbresp}", level='INFO')
        # log_check_pb_status(pbresp, "Vacuum ON")

        pbresp = pb._set_chuck_mode(interlock=True) # turn on the interlock
        log_utils.log_status(hw_module, f"Interlock ON: {pbresp}", level='INFO')

        pbresp = pb._move_chuckZ_separation() # move to separation position
        log_utils.log_status(hw_module, f"Moved to separation position: {pbresp}", level='INFO')
        # log_check_pb_status(pbresp, "Moved to separation position")

        if not manual_mode:
            # move to home position
            pbresp = pb._move_chuckXY_micron(dx = 0, dy = 0, posref='H') # move to home position
            log_utils.log_status(hw_module, f"Moved to home position: {pbresp}", level='INFO')
        else:
            # skip moving to home position
            log_utils.log_status(hw_module, f"Manual mode enabled, skipping moving to home position", level='INFO')

        for structure in filtered_df.iterrows(): # iterate over all structures
            stop_flag = sys_config.TEMP_STOP_FLAG_PATH # check for the stop flag
            if os.path.exists(stop_flag): # if the stop flag exists, raise an exception to stop the measurement
                raise Exception("Stop flag detected, stopping the measurement")
            
            # get the structure details
            structure_counter += 1
            curr_variation_item = str(structure[1][curr_variation]) # get the material type of this structure
            block = structure[1]['block'] # get the block number of this structure
            die = structure[1]['die'] # get the die number of this structure
            pin_variation_item = str(structure[1][pin_variation])
            block_row = structure[1]['block_row']
            block_col = structure[1]['block_column']
            block_idx_in_die = structure[1]['block_idx_in_die']

            hw_module = 'PB'
            pbresp = pb._move_chuckZ_separation() # move to separation position
            log_utils.log_status(hw_module, f"Moved to separation position: {pbresp}", level='INFO')

            if not manual_mode: # move to the structure position
                pbresp = pb._move_chuckXY_micron(dx = structure[1]['coords_Href'][0], dy = structure[1]['coords_Href'][1], posref='H')
                log_utils.log_status(hw_module, f"Moved to structure position: {pbresp}", level='INFO')
            else: # skip moving to the structure position
                log_utils.log_status(hw_module, f"Manual mode enabled, skipping moving to structure position", level='INFO')

            pbresp = pb._move_chuckZ_contact() # move to contact position
            log_utils.log_status(hw_module, f"Moved to contact position: {pbresp}", level='INFO')


            selected_cfg_map = cfg_map[pin_variation_item] # get the configuration map for the current structure
            for i in range(repeat): # repeat the measurement for each structure
                if stb_flag: # if stability test is enabled, output the stability measurement number
                    print(f"Executing stability measurement {i+1}/{repeat}")

                event_counter += 1
                for layout_num, sub_item in selected_cfg_map.items(): # iterate over the configuration map
                    stop_flag = sys_config.TEMP_STOP_FLAG_PATH # check for the stop flag
                    if os.path.exists(stop_flag): # if the stop flag exists, raise an exception to stop the measurement
                        raise Exception("Stop flag detected, stopping the measurement")

                    alias = sub_item.get('alias', 'default_alias') # get the alias for the current configuration

                    selected_current_config = sub_item.get('curr_cfg', None) # get the current configuration for the current structure
                    curr_cfg = curr_cfgs[selected_current_config] # get the current configuration for the current structure

                    force_currents = curr_cfg[curr_variation_item] # get the force currents for the current structure

                    selected_pin_configs = sub_item.get('pin_cfgs', []) # get the pin configurations for the current structure
                    selected_pin_configs = {key: pin_cfgs[key] for key in selected_pin_configs} # filter the pin configurations

                    # get the measurement parameters
                    comp_v = sub_item.get('comp_v', 1.0)
                    nplc = sub_item.get('nplc', 5.0)
                    nrdgs = sub_item.get('nrdgs', 1)

                    for pin_config_name, pin_config in selected_pin_configs.items():# iterate over the pin configurations
                        
                        smu_ch_list = get_channels(pin_config, smu_channel_map) # get the SMU channel for the pin configuration
                        if not smu_ch_list: # if no SMU channel found, skip the measurement
                            print(f"No SMU channel found for the pin configuration --- {pin_config_name}, skipping...")
                            continue
                        elif len(smu_ch_list) > 1: # if multiple SMU channels found, skip the measurement
                            print(f"Multiple SMU channels found for the pin configuration --- {pin_config_name}, skipping...")
                            continue
                        smu_ch = smu_ch_list[0] # get the SMU channel NOTE: SUPPORTS ONLY ONE CHANNEL

                        # parse the pin configuration name
                        if pin_config_name[-1] == 'P' or pin_config_name[-1] == 'N':
                            base_config_name = pin_config_name[:-2]
                            config_pol = pin_config_name[-1]
                        else:
                            base_config_name = pin_config_name
                            config_pol = 'P'

                        # make port-pin connections
                        hw_module = 'SWM'
                        matrix.connect_ports_pins(pin_map=pin_config)
                        log_utils.log_status(hw_module, f"Connected the pins for {pin_config_name} with pin map: {pin_config}", level='INFO')
                        matrix.check_status()
                        
                        # prepare the structure configuration information
                        structure_cfg_info = {
                            "die": die,
                            "block": block,
                            "pin_variation": pin_variation_item, # for agcl design: structure
                            "curr_variation": curr_variation_item, # for agcl design: material
                            "pin_config": pin_config_name,
                        }

                        for current in force_currents: # iterate over the force currents
                            smu_comped = False
                            # current_source = current
                            #measure the voltage for positive and negative current
                            pos_meanv, pos_meanI, comped_event, _ = forceI_measureV(smu=smu, dmm=dmm, smu_channel=smu_ch, force_current=current, nrdgs=nrdgs, nplc=nplc, structure_cfg_info=structure_cfg_info, writer=writer, volt_comp=comp_v)
                            smu_comped = smu_comped or comped_event

                            neg_meanv, neg_meanI, comped_event, _ = forceI_measureV(smu=smu, dmm=dmm, smu_channel=smu_ch, force_current=-current, nrdgs=nrdgs, nplc=nplc, structure_cfg_info=structure_cfg_info, writer=writer, volt_comp=comp_v)
                            smu_comped = smu_comped or comped_event

                            # cancel offset
                            cal_V = (pos_meanv - neg_meanv) / 2
                            cal_I = (pos_meanI - neg_meanI) / 2
                            # record the results
                            results_df = results_df._append({
                                'wafer_idx': wafer_idx,
                                'die': die,
                                'block': block,
                                'block_idx_in_die': block_idx_in_die,
                                pin_variation: pin_variation_item,
                                curr_variation: curr_variation_item,
                                'alias': alias,
                                'pin_config': base_config_name,
                                'pol': config_pol,
                                'force_current': current,
                                'meanV': cal_V,
                                'meanI': cal_I,
                                'hit_comp': smu_comped,
                                'block_row': block_row,
                                'block_column': block_col,
                                'event_counter': event_counter,
                                '__pin_variation_object': pin_variation,
                                '__curr_variation_object': curr_variation
                            }, ignore_index=True)
                            # new_row = pd.DataFrame([{
                            #     'die': die,
                            #     'block': block,
                            #     pin_variation: pin_variation_item,
                            #     curr_variation: curr_variation_item,
                            #     'alias': alias,
                            #     'pin_config': base_config_name,
                            #     'pol': config_pol,
                            #     'force_current': current,
                            #     'meanV': cal_V,
                            #     'meanI': cal_I,
                            #     'hit_comp': smu_comped,
                            #     'block_row': block_row,
                            #     'block_column': block_col
                            # }])

                            # results_df = pd.concat([results_df, new_row], ignore_index=True)

            hw_module = 'PB'
            pb._move_chuckZ_separation() # move to separation position
            log_utils.log_status(hw_module, f"Moved to separation position: {pbresp}", level='INFO')
            if not stb_flag:
                print(f"{structure_counter}/{total_structures} structure tested")

        raise_error = False # set the raise error flag to False

        # cleanup
        hw_module = 'DMM'
        dmm._beep_once() # beep once
        dmm.check_errors() # check for DMM errors
        
        hw_module = 'SMU'
        smu.check_errors() # check for SMU errors
        log_utils.log_status(module_name='DMM', status_message='Beeped once', level='INFO')
        # record the completion of the measurement
        log_utils.log_status(module_name='General', status_message='Measurement completed', level='INFO')

    except Exception as e:
        # handle the exception and cleanup
        try:
            # pb._move_chuck_load()
            hw_module = 'PB'
            pb._move_chuckZ_separation() # move to separation position
            log_utils.log_status(hw_module, f"Exception Cleanup: Moved to separation position: {pbresp}", level='INFO')
        except Exception as cleanup_error:
            handle_exception(cleanup_error, hw_module, raise_error=False, additional_info='Error during cleanup after main exception occured')

        try:
            hw_module = 'SMU'
            smu._reset() # reset smu this will also turn off the channel
            log_utils.log_status(hw_module, f"Exception Cleanup: SMU reset", level='INFO')
            smu.query_translate_error_list() # clear the error list
            log_utils.log_status(hw_module, f"Exception Cleanup: SMU error list cleared", level='INFO')
        except Exception as cleanup_error:
            handle_exception(cleanup_error, hw_module, raise_error=False, additional_info='Error during cleanup after main exception occured')

        try:
            hw_module = 'SWM'
            matrix._clear_matrix() # clear the matrix
            log_utils.log_status(hw_module, f"Exception Cleanup: Disconnecting all pins", level='INFO')

        except Exception as cleanup_error:
            handle_exception(cleanup_error, hw_module, raise_error=False, additional_info='Error during cleanup after main exception occured')

        try:
            hw_module = 'DMM'
            dmm._set_display('ON') # turn on the display
            log_utils.log_status(hw_module, f"Exception Cleanup: DMM display ON", level='INFO')
            dmm.query_error_list() # clear error list
            log_utils.log_status(hw_module, f"Exception Cleanup: DMM display ON and error list cleared", level='INFO')

        except Exception as cleanup_error:
            handle_exception(cleanup_error, hw_module, raise_error=False, additional_info='Error during cleanup after main exception occured')

        exception_occurred = handle_exception(e, hw_module, raise_error) # mainly for error logging

    finally:
        # return df and close the file anyway
        measurements.close()
        return results_df, event_name, exception_occurred
    
if __name__ == "__main__":
    # Back end script for running the measurement

    import config as sys_config # import the system configuration

    def str_to_bool(value):
        # Function to convert string to boolean
        if isinstance(value, bool):
            return value
        if value.lower() in ('true', '1', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # parse the arguments
    parser = argparse.ArgumentParser(description='Run measurement script with parameters.')
    parser.add_argument('--filtered_df', type=str, required=True, help='Path to the filtered DataFrame (Pickle file)')
    parser.add_argument('--pin_cfgs', type=str, required=True, help='Path to the pin configurations (JSON file)')
    parser.add_argument('--cfg_map', type=str, required=True, help='Path to the configuration map (JSON file)')
    parser.add_argument('--curr_cfgs', type=str, required=True, help='Path to the current configurations (JSON file)')
    parser.add_argument('--smu_channel_map', type=str, required=True, help='Path to the SMU channel map (JSON file)')
    parser.add_argument('--repeat', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--stability_test', type=str_to_bool, default=False, help='Run stability test')
    parser.add_argument('--manual_mode', type=str_to_bool, default=False, help='Run in manual mode')

    args = parser.parse_args()

    # load the data
    with open(args.filtered_df, 'rb') as f:
        filtered_df = pickle.load(f)
    with open(args.pin_cfgs, 'r') as f:
        pin_cfgs = json.load(f)
    with open(args.cfg_map, 'r') as f:
        cfg_map = json.load(f)
    with open(args.curr_cfgs, 'r') as f:
        curr_cfgs = json.load(f)
    with open(args.smu_channel_map, 'r') as f:
        smu_channel_map = json.load(f)

    # smu_channel_map = sys_config.SMU_CHANNEL_MAP

    pb_addr = sys_config.PB_GPIB_ADDR
    smu_addr = sys_config.SMU_GPIB_ADDR
    swm_addr = sys_config.SWM_GPIB_ADDR
    dmm_addr = sys_config.DMM_GPIB_ADDR

    # initialize the hardware
    rm, pb, dmm, smu, matrix = hw_init(
        pb_addr=pb_addr, 
        smu_addr=smu_addr, 
        swm_addr=swm_addr, 
        dmm_addr=dmm_addr
    ) 

    # call the main measurement function
    results_df, event_name, exception_occurred = main_measurement(
        pb=pb, dmm=dmm, smu=smu, matrix=matrix,
        filtered_df=filtered_df,
        pin_cfgs=pin_cfgs,
        cfg_map=cfg_map,
        curr_cfgs=curr_cfgs,
        smu_channel_map=smu_channel_map,
        repeat=args.repeat,
        manual_mode=args.manual_mode,
        stb_flag=args.stability_test,
        raw_measurement_output_folder = sys_config.RAW_MEASUREMENT_OUTPUT_FOLDER
    )

    rm.close()  # close the resource manager, release the hardware resources
    # remove the stop flag file if it exists

    
    # to avoid the error of not finding the temp folder for any reason
    temp_folder = sys_config.TEMP_FOLDER_PATH
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    stop_flag = sys_config.TEMP_STOP_FLAG_PATH

    if os.path.exists(stop_flag):
        os.remove(stop_flag)

    # save the results to a file
    if not args.stability_test:
        ## absolute path
        # result_folder = os.path.join(os.getcwd(), f'{sys_config.GROUPED_MEASUREMENTS_FOLDER}/{filtered_df["design"].iloc[0]}')
        ## relative path
        result_folder = f'{sys_config.GROUPED_MEASUREMENTS_FOLDER}/{filtered_df["design"].iloc[0]}'
        # result_folder = os.path.join(sys_config.GROUPED_MEASUREMENTS_FOLDER, filtered_df['design'].iloc[0])
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        file_path = os.path.join(result_folder, f'{event_name}_results.csv')
        status_file_path = sys_config.STATUS_FILE_PATH # status file path for the wafer map test
    else:
        ## absolute path
        # result_folder = os.path.join(os.getcwd(), f'{sys_config.STABILITY_MEASUREMENT_FOLDER}/{filtered_df["design"].iloc[0]}')
        ## relative path
        result_folder = f'{sys_config.STABILITY_MEASUREMENT_FOLDER}/{filtered_df["design"].iloc[0]}'
        # result_folder = os.path.join(sys_config.STABILITY_MEASUREMENT_FOLDER, filtered_df['design'].iloc[0])
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        file_path = os.path.join(result_folder, f'{event_name}_stability_results.csv')
        status_file_path = sys_config.STABILITY_STATUS_FILE_PATH # status file path for the stability test

    # create a status file
    stu = {}
    stu['event_name'] = event_name
    stu['file_path'] = file_path
    if exception_occurred is not None:

        stu['exception_occurred'] = f"{type(exception_occurred)}, {exception_occurred}, {exception_occurred.__traceback__}"
    else:
        stu['exception_occurred'] = None

    # save_file = 'temp/stu.json'
    # save_file = sys_config.STATUS_FILE_PATH
    with open(status_file_path, 'w') as f:
        json.dump(stu, f)

    results_df.to_csv(file_path, index=False)
    # main_measurement()
