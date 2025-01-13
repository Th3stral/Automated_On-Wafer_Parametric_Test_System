import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import argparse
import json

import pyvisa
from utils.hw_utils import SMUinterface, DMMinterface, PBinterface, SWMinterface, HardwareError
from utils.mea_utils import handle_exception
import utils.log_utils as log_utils

import config as sys_config
import datetime

def hw_stu_check(pb_addr=1, smu_addr=23, swm_addr=22, dmm_addr=12):
    """
    Performs health checks on PB, DMM, SMU, and SWM hardware components.

    Args:
        pb_addr (int): Address for the Prober (PB).
        smu_addr (int): Address for the Source Measure Unit (SMU).
        swm_addr (int): Address for the Switch Matrix (SWM).
        dmm_addr (int): Address for the Digital Multimeter (DMM).

    Returns:
        dict: A dictionary containing the health check results for each device.
    """
    stu_dict = {}
    rm = pyvisa.ResourceManager() # create the resource manager osbject

    stu_dict['PB'] = {'passed': False, 'message': None} # create a dictionary for the prober
    try:
        pb = PBinterface(rm, addr=pb_addr) # create an instance of the prober
        pb.driver.timeout = 1000 # set the timeout for the prober
        pb.delay_time = 0.1 # set the delay time for the prober
        pbstu = pb._set_chuck_vacuum(vacuum_on=True) # turn on the vacuum
        # pbstu = pb._move_chuckZ_separation()
        if pbstu != '0  05 1': # check if the correct status is returned
            pb_check_passed = False
            pbstu = f'Vacuum ON failed with returned status: {pbstu}' # if not, set the message
        else:
            pb_check_passed = True
            pbstu = 'Normal' # if yes, set the message
    except Exception as e:
        # if an exception is raised, set the check to failed and set the message to the exception
        pb_check_passed = False
        pbstu = str(e)
    stu_dict['PB']['passed'] = pb_check_passed
    stu_dict['PB']['message'] = pbstu

    stu_dict['DMM'] = {'passed': False, 'message': None} # create a dictionary for the DMM
    try:
        # create an instance of the DMM
        dmm = DMMinterface(rm, addr=dmm_addr)
        dmm.driver.timeout = 1000# set the timeout for the DMM
        dmm_id = dmm._query_identity()
        if dmm_id != 'HP3458A': # check if the correct id is returned
            dmm_check_passed = False
            dmmstu = f'idcheck failed with returned id: {dmm_id}'
        else:
            dmm.check_errors()
            dmm_check_passed = True
            dmmstu = 'Normal'
    except Exception as e:
        # if an exception is raised, set the check to failed and set the message to the exception
        dmm_check_passed = False
        dmmstu = str(e)
    stu_dict['DMM']['passed'] = dmm_check_passed
    stu_dict['DMM']['message'] = dmmstu

    stu_dict['SMU'] = {'passed': False, 'message': None} # create a dictionary for the SMU
    try:
        # create an instance of the SMU
        smu = SMUinterface(rm, addr=smu_addr)
        smu.driver.timeout = 1000 # set the timeout for the SMU
        smu_id = smu._query_identity()
        if 'HEWLETT PACKARD,4142B' not in smu_id: # check if the correct id is returned
            smu_check_passed = False
            smustu = f'idcheck failed with returned id: {smu_id}'
        else:
            smu.check_errors()
            smu_check_passed = True
            smustu = 'Normal'
    except Exception as e:
        # if an exception is raised, set the check to failed and set the message to the exception
        smu_check_passed = False
        smustu = str(e)
    stu_dict['SMU']['passed'] = smu_check_passed
    stu_dict['SMU']['message'] = smustu

    stu_dict['SWM'] = {'passed': False, 'message': None}
    try:
        # create an instance of the SWM
        matrix = SWMinterface(rm, addr=swm_addr)
        matrix.driver.timeout = 1000
        swm_id = matrix._query_identity()
        if 'HP 4084B' not in swm_id: # check if the correct id is returned
            matrix_check_passed = False
            matrixstu = f'idcheck failed with returned id: {swm_id}'
        else:
            matrixstu = matrix.check_status()
            if not (matrixstu & 0b00000001):
                matrix_check_passed = False
                matrixstu = f'SWM is not in ready state with status: {matrixstu}'
            else:
                matrix_check_passed = True
                matrixstu = 'Normal'
    except Exception as e:
        matrix_check_passed = False
        matrixstu = str(e)
    stu_dict['SWM']['passed'] = matrix_check_passed
    stu_dict['SWM']['message'] = matrixstu

    rm.close() #release the resource manager

    return stu_dict

def main_cal(check_dict = {}):
    # calibrate the DMM and SMU
    log_utils.configure_logger(log_filename='hardware_operations.log', level='INFO') # configure the logger
    rm = pyvisa.ResourceManager()
    exception_occurred = {}
    try:
        # get the checklist for the DMM and SMU
        dmm_checklist = check_dict.get('DMM', None)

        smu_checklist = check_dict.get('SMU', None)
        dmm_cal = dmm_checklist.get('acal', None)
        smu_cal = smu_checklist.get('acal', False)
        dmm_test = dmm_checklist.get('selftest', False)
        smu_test = smu_checklist.get('selftest', False)
        total_checks = [dmm_cal, smu_cal, dmm_test, smu_test] # create a list of the checks to be performed
        # check the number of trues or strings in the checklist
        tasks_to_perform = total_checks.count(True) + total_checks.count('DCV') + total_checks.count('ALL') # count the number of tasks to be performed
        task_performed = 0

        hw_module = 'DMM'
        # if dmm_checklist:
        if dmm_cal or dmm_test:
            dmm = DMMinterface(rm, addr=12) # create an instance of the DMM
            if dmm_checklist.get('selftest', False):
                print('Performing self test for DMM...')
                dmm.perform_self_test() # perform the self test
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

            cal_type = dmm_checklist.get('acal', None)
            if cal_type:
                print(f'Performing self calibration {cal_type} for DMM...')
                dmm.perform_self_calibration(cal_type) # perform the self calibration
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

    except Exception as e:
        # exception_occurred = handle_exception(e, hw_module=hw_module, raise_exception=True)
        exception_occurred['DMM'] = (handle_exception(e, hw_module=hw_module, raise_exception=True))

    try:    
        hw_module = 'SMU'
        # if smu_checklist:
        if smu_cal or smu_test: # check if the SMU calibration or self test is to be performed
            smu = SMUinterface(rm, addr=23) # create an instance of the SMU
            if smu_checklist.get('selftest', False): # check if the self test is to be performed
                print('Performing self test for SMU...')
                smu.perform_self_test()
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

            if smu_checklist.get('acal', False): # check if the self calibration is to be performed
                print('Performing self calibration for SMU...')
                smu.perform_self_calibration()
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

    except Exception as e:
        # exception_occurred = handle_exception(e, hw_module=hw_module, raise_exception=True)
        exception_occurred['SMU'] = (handle_exception(e, hw_module=hw_module, raise_exception=True))

    finally:
        rm.close() # release the resource manager
        print('All tasks completed') # print a message that all tasks have been completed
        return exception_occurred


if __name__ == '__main__':

    # back end script to run the self calibration and self test
    #parse the arguments
    parser = argparse.ArgumentParser(description='Run measurement script with parameters.')
    # parser.add_argument('--dmm_acal', type=str, default=None, help='DMM self calibration type')
    # parser.add_argument('--smu_acal', type=bool, default=False, help='SMU self calibration')
    # parser.add_argument('--dmm_selftest', type=bool, default=False, help='DMM self test')
    # parser.add_argument('--smu_selftest', type=bool, default=False, help='SMU self test')
    parser.add_argument('--check_dict', type=str, default=None, help='Dictionary containing the checks to be performed')

    args = parser.parse_args()
    with open(args.check_dict, 'r') as f:
        check_dict = json.load(f)


    # initialize the dictionary to store the results of the self calibration and self test        
    exception_occurred = main_cal(check_dict)    
    cal_stu = {}
    if not exception_occurred:
        cal_stu['status'] = 'Self-Calibration/Self-Test successfully completed'
        cal_stu['exception'] = None
        cal_stu['tasks'] = check_dict
        cal_stu['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        cal_stu['status'] = 'Self-Calibration/Self-Test failed'
        cal_stu['exception'] = ['{}: {}, errors: {}'.format(k, v, getattr(v, 'errors', None)) for k, v in exception_occurred.items()]
        cal_stu['tasks'] = check_dict
        cal_stu['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    temp_folder = sys_config.TEMP_FOLDER_PATH
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    # save_file = 'temp/cal_stu.json'
    save_file = sys_config.CAL_STATUS_FILE_PATH
    with open(save_file, 'w') as f:
        json.dump(cal_stu, f)

    
        
                

                
        
    
