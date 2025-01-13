import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import argparse
import json

import pyvisa
from utils.hw_utils import SMUinterface, DMMinterface, PBinterface, SWMinterface, HardwareError

import config as sys_config


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

if __name__ == '__main__':
    stu_dict = hw_stu_check()
    temp_folder = sys_config.TEMP_FOLDER_PATH
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    save_file = 'temp/hw_stu.json'
    with open(save_file, 'w') as f:
        json.dump(stu_dict, f)