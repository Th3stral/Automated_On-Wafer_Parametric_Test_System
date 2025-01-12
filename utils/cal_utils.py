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

    stu_dict = {}
    rm = pyvisa.ResourceManager() # create the resource manager osbject

    stu_dict['PB'] = {'passed': False, 'message': None}
    try:
        pb = PBinterface(rm, addr=pb_addr)
        pb.driver.timeout = 1000
        pb.delay_time = 0.1
        pbstu = pb._set_chuck_vacuum(vacuum_on=True) # turn on the vacuum
        # pbstu = pb._move_chuckZ_separation()
        if pbstu != '0  05 1':
            pb_check_passed = False
            pbstu = f'Vacuum ON failed with returned status: {pbstu}'
        else:
            pb_check_passed = True
            pbstu = 'Normal'
    except Exception as e:
        pb_check_passed = False
        pbstu = str(e)
    stu_dict['PB']['passed'] = pb_check_passed
    stu_dict['PB']['message'] = pbstu

    stu_dict['DMM'] = {'passed': False, 'message': None}
    try:
        dmm = DMMinterface(rm, addr=dmm_addr)
        dmm.driver.timeout = 1000
        dmm_id = dmm._query_identity()
        if dmm_id != 'HP3458A':
            dmm_check_passed = False
            dmmstu = f'idcheck failed with returned id: {dmm_id}'
        else:
            dmm.check_errors()
            dmm_check_passed = True
            dmmstu = 'Normal'
    except Exception as e:
        dmm_check_passed = False
        dmmstu = str(e)
    stu_dict['DMM']['passed'] = dmm_check_passed
    stu_dict['DMM']['message'] = dmmstu

    stu_dict['SMU'] = {'passed': False, 'message': None}
    try:
        smu = SMUinterface(rm, addr=smu_addr)
        smu.driver.timeout = 1000
        smu_id = smu._query_identity()
        if 'HEWLETT PACKARD,4142B' not in smu_id:
            smu_check_passed = False
            smustu = f'idcheck failed with returned id: {smu_id}'
        else:
            smu.check_errors()
            smu_check_passed = True
            smustu = 'Normal'
    except Exception as e:
        smu_check_passed = False
        smustu = str(e)
    stu_dict['SMU']['passed'] = smu_check_passed
    stu_dict['SMU']['message'] = smustu

    stu_dict['SWM'] = {'passed': False, 'message': None}
    try:
        matrix = SWMinterface(rm, addr=swm_addr)
        matrix.driver.timeout = 1000
        swm_id = matrix._query_identity()
        if 'HP 4084B' not in swm_id:
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

    rm.close()

    return stu_dict

def main_cal(check_dict = {}):
    log_utils.configure_logger(log_filename='hardware_operations.log', level='INFO') # configure the logger
    rm = pyvisa.ResourceManager()
    exception_occurred = {}
    try:
        dmm_checklist = check_dict.get('DMM', None)

        smu_checklist = check_dict.get('SMU', None)
        dmm_cal = dmm_checklist.get('acal', None)
        smu_cal = smu_checklist.get('acal', False)
        dmm_test = dmm_checklist.get('selftest', False)
        smu_test = smu_checklist.get('selftest', False)
        total_checks = [dmm_cal, smu_cal, dmm_test, smu_test]
        # check the number of trues or strings in the checklist
        tasks_to_perform = total_checks.count(True) + total_checks.count('DCV') + total_checks.count('ALL')
        task_performed = 0

        hw_module = 'DMM'
        # if dmm_checklist:
        if dmm_cal or dmm_test:
            dmm = DMMinterface(rm, addr=12)
            if dmm_checklist.get('selftest', False):
                print('Performing self test for DMM...')
                dmm.perform_self_test()
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

            cal_type = dmm_checklist.get('acal', None)
            if cal_type:
                print(f'Performing self calibration {cal_type} for DMM...')
                dmm.perform_self_calibration(cal_type)
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

    except Exception as e:
        exception_occurred['DMM'] = (handle_exception(e, hw_module=hw_module, raise_exception=True))

    try:    
        hw_module = 'SMU'
        # if smu_checklist:
        if smu_cal or smu_test:
            smu = SMUinterface(rm, addr=23)
            if smu_checklist.get('selftest', False):
                print('Performing self test for SMU...')
                smu.perform_self_test()
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

            if smu_checklist.get('acal', False):
                print('Performing self calibration for SMU...')
                smu.perform_self_calibration()
                task_performed += 1
                print(f'{task_performed} out of {tasks_to_perform} tasks performed')

    except Exception as e:
        # exception_occurred = handle_exception(e, hw_module=hw_module, raise_exception=True)
        exception_occurred['SMU'] = (handle_exception(e, hw_module=hw_module, raise_exception=True))

    finally:
        rm.close()
        print('All tasks completed')
        return exception_occurred


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Run measurement script with parameters.')
    # parser.add_argument('--dmm_acal', type=str, default=None, help='DMM self calibration type')
    # parser.add_argument('--smu_acal', type=bool, default=False, help='SMU self calibration')
    # parser.add_argument('--dmm_selftest', type=bool, default=False, help='DMM self test')
    # parser.add_argument('--smu_selftest', type=bool, default=False, help='SMU self test')
    parser.add_argument('--check_dict', type=str, default=None, help='Dictionary containing the checks to be performed')

    args = parser.parse_args()
    with open(args.check_dict, 'r') as f:
        check_dict = json.load(f)


        
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

    
        
                

                
        
    
