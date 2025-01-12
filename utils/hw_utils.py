import time
from struct import unpack
import json
import functools
import numpy as np
from types import NoneType

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))




class HardwareError(Exception):
    """custom exception for hardware errors"""
    def __init__(self, message, errors=[], module=None):
        super().__init__(message)
        self.errors = errors
        self.module = module


class SMUhwcmd:

    def __init__(self, rm, addr=23):
        self.rm = rm # reference to Resource manager
        self.driver = self.rm.open_resource(f'GPIB0::{addr}::INSTR')
        self.driver.timeout = 20000
        self.driver.write_termination = '\r\n'
        self.driver.read_termination = '\r\n'

    def _zero_output(self, ch):
        # this command sets the specific channel to Zero output
        self.driver.write(f'DZ{ch}')
    
    def _channel_off(self, ch):
        # this command sets the specific channel output to off
        self.driver.write(f'CL{ch}')
    
    def _channel_on(self, ch):
        # this command sets the specific channel output to on
        self.driver.write(f'CN{ch}')

    def _set_force_current(self, ch=1, current_output_range=0, current_value=0, volt_comp=1, comp_polar_mode=0):
        """
        ch: channel number
            {
            1: smu1(MPSMU),
            3: smu2(HPSMU),
            4: smu3(MPSMU),
            5: smu4(MPSMU),
            6: VS/VMU
            }

        current_output_range:   for smu1,3,4(MPSMU) range: 0, 11-19
                                for smu2(HPSMU) range: 0, 11-20
                                0: auto ranging
                                11: 1nA limited auto ranging 
                                12: 10nA limited auto ranging
                                13: 100nA limited auto ranging
                                14: 1uA limited auto ranging
                                15: 10uA limited auto ranging
                                16: 100uA limited auto ranging
                                17: 1mA limited auto ranging
                                18: 10mA limited auto ranging
                                19: 100mA limited auto ranging
                                20: 1A limited auto ranging

        current_value: 
            numeric value in A
            for smu1,3,4(MPSMU) range: -100mA to 100mA (-100E-3 to 100E-3)
            for smu2(HPSMU) range: -1A to 1A
        volt_comp: numeric value in V, allowed range see manual
        comp_polar_mode: 0: auto, 1: manual
        """

        # if current_output_range != 0 and (current_output_range < 11 or current_output_range > 20):
        #     raise ValueError("Invalid current output range")
        # if not ch in [1, 3, 4, 5, 6, '1', '3', '4', '5', '6']:
        #     raise ValueError("Invalid channel number, valid channel numbers are 1, 3, 4, 5, 6 (updated on Dec 2024)")
        # if not isinstance(current_value, (int, float, np.integer, np.floating)):
        #     raise TypeError("Invalid current value, current value must be an integer or float")
        # if not isinstance(volt_comp, (int, float, np.integer, np.floating, NoneType)):
        #     raise TypeError("Invalid voltage compensation value, voltage compensation value must be an integer or float")
        # if not comp_polar_mode in [0, 1, None]:  
        #     raise ValueError("Invalid compensation polar mode, compensation polar mode must be 0, 1 or None")
        
        cmd = f'DI{ch},{current_output_range},{current_value}'
        if volt_comp != None:
            cmd += f',{volt_comp}'
        if comp_polar_mode != None:
            cmd += f',{comp_polar_mode}'
        self.driver.write(cmd)
    
    def _measure_spot_current(self, ch):
        # take spot current measurement here so query can be used
        """TODO: check if query ascii can be used"""
        current_with_header = self.driver.query(f'TI{ch}')
        return current_with_header
    
    def _query_errors(self):
        """
        according to the error code chapter of HP-IB command manual, 
        HP4142B can only store the first four errors in error register:
        [first error code, second error code, third error code, fourth error code]
        """
        fetched_errors = self.driver.query_ascii_values('ERR?', converter='s')
        return fetched_errors
    
    def _reset(self):
        self.driver.write('*RST')

    def _query_identity(self):
        return self.driver.query('*IDN?')
    
    def _start_self_calibration(self, slot=None):
        self.driver.write(f'CA {slot or ""}')

    def _start_self_test(self, slot=None):
        test_result = self.driver.query(f"*TST? {slot or ''}")
        return test_result
    

class SMUinterface(SMUhwcmd):
    # according to 4142B HP-IB Command Reference Manual
    def __init__(self, rm, addr=23, error_code_path = 'utils/error_code_files/smu_error_codes.json'):
        # SMUhwcmd.__init__(self, rm, addr)
        super().__init__(rm, addr)
        try:
            with open (error_code_path, 'r') as file:
            # with open('smu_error_codes.json', 'r') as file:
                self.error_codes = json.load(file)
                self.error_translation = True
        except Exception as e:
            self.error_codes = {}
            self.error_translation = False
            print(f"Error translation function not available: {e}")

    def __search_error_dict(self, error_code, error_codes):
        return error_codes.get(error_code, f"Error code {error_code} not found. Please check the manual for more information.")

    def __translate_error_message(self, error_code):
        if int(error_code) <= 999:
            error_message = self.__search_error_dict(error_code, self.error_codes)
            return f"SMU error: {error_message}"
        elif int(error_code) <= 9999:
            slot = error_code[:1]
            error_code = error_code[1:]
            error_message = self.__search_error_dict(error_code, self.error_codes)
            return f"SMU slot {slot} error: {error_message}"
        else:
            channel = error_code[:2]
            error_code = error_code[2:]
            print(error_code, channel)
            error_message = self.__search_error_dict(error_code, self.error_codes)
            return f"VS/VMU channel {channel} error: {error_message}"    

    def query_translate_error_list(self):
        fetched_errors = self._query_errors()
        # no error
        if fetched_errors[0] == "0":
            return []
        # error exists
        else:
            # if self.error_codes != {}:
            for i, error_code in enumerate(fetched_errors):
                if (error_code != "0"):
                    if self.error_translation:
                        fetched_errors[i] = (error_code, self.__translate_error_message(error_code))
                    else:
                        fetched_errors[i] = (error_code, f"Error translation disabled. Original error code: {error_code}")
                else:
                    fetched_errors = fetched_errors[:i]
                    break
            return fetched_errors
        
    def check_errors(self):
        error_list = self.query_translate_error_list()
        if error_list:
            raise HardwareError(message = f"Error from SMU: {error_list}", errors = error_list, module = "SMU")

    def get_spot_current(self, ch):  
        # take spot current measurement here so query can be used
        current_with_header = self._measure_spot_current(ch)
        current_value = float(current_with_header[3:])
        status = current_with_header[0]
        channel = current_with_header[1]
        measurement_type = current_with_header[2]
        return current_value, status, channel, measurement_type
    
    def perform_self_calibration(self, slot=None):
        """takes about 25s to complete"""
        original_timeout = self.driver.timeout
        
        try:
            self.driver.timeout = 75000 # 75s timeout for self test
            self._start_self_calibration(slot)
            self.check_errors()
        except HardwareError as e:
            self.driver.timeout = 20000 # reset timeout
            raise HardwareError(message = f"SMU self calibration failed", errors = e.errors, module = "SMU")
        finally:
            self.driver.timeout = original_timeout
    
    def perform_self_test(self, slot=None):
        """takes about 30s to complete for a pass event"""
        TIMEOUT_SELF_TEST = 100000  # Timeout for self-test
        BIT_TO_ERROR_MAPPING = {
            0b000000000001: "Slot 1 unit failed",
            0b000000000010: "Slot 2 unit failed",
            0b000000000100: "Slot 3 unit failed",
            0b000000001000: "Slot 4 unit failed",
            0b000000010000: "Slot 5 unit failed",
            0b000000100000: "Slot 6 unit failed",
            0b000001000000: "Slot 7 unit failed",
            0b000010000000: "Slot 8 unit failed",
            0b000100000000: "Mainframe failed",
            0b001000000000: (
                "Did not perform Self-test on one or more HVUs. "
                "Short the INTLK terminal and perform Self-test again."
            ),
            0b010000000000: (
                "Did not perform Self-test on one or more HVUs, and lost the Self-calibration "
                "data of the HVU(s). Short the INTLK terminal and perform Self-test again."
            ),
        }

        original_timeout = self.driver.timeout
        try:
            self.driver.timeout = TIMEOUT_SELF_TEST
            test_result = self._start_self_test(slot)
            if test_result != '0':
                try:
                    test_result = int(test_result)
                    translated_result = [
                        message for bit, message in BIT_TO_ERROR_MAPPING.items() if test_result & bit
                    ]
                except ValueError:
                    translated_result = [f"Unknown test result: {test_result}"]
                raise HardwareError(
                    message=(
                        f"SMU self test failed for slot {slot} with result: {test_result}. "
                        "Please check the manual for more information."
                    ),
                    errors=translated_result,
                    module="SMU"
                )
        finally:
            self.driver.timeout = original_timeout
    
class DMMhwcmd:
    def __init__(self, rm, addr=12):
        self.rm = rm
        self.driver = self.rm.open_resource(f'GPIB0::{addr}::INSTR')
        self.driver.timeout = 20000
        self.driver.write_termination = '\r\n'
        self.driver.read_termination = '\r\n'


    # @property
    # def dmm_mem_format(self):
    #     return self._dmm_mem_format
    
    def _apply_preset(self, preset):
        """
        Configures the multimeter to one of three predefined states:

        NORM---PRESET NORM is similar to RESET but optimizes the multimeter for remote operation.
        FAST---PRESET FAST configures the multimeter for fast readings, fast transfer to memory
                , and fast transfer from memory to GPIB
        DIG ---PRESET DIG configures the multimeter for DCV digitizing.

        DMM Power on type: not applicable
        DMM default type (when command = PRESET): NORM
        """

        if preset not in ['NORM', 'FAST', 'DIG', '']:
            raise ValueError('Invalid input: preset must be NORM, FAST, or DIG')
        self.driver.write(f'PRESET {preset or ""}')

    def _set_mem_format(self, format):
        """
        this command clears the reading memory and designates the storage format for
        new readings.
        following are the memory formats that can be specified:

            ASCII---ASCII
                    -16 bytes per reading
                    (it is actually 15 bytes for the reading plus 1 byte for a null character 
                    which is used to separatestored ASCII readings only)
            SINT ---Single Integer
                    -16 bits 2's complement (2 bytes per reading)
            DINT ---Double Integer
                    -32 bits 2's complement (4 bytes per reading)
            SREAL---Single Real
                    -(IEEE-754) 32 bits, (4 bytes per reading)
            DREAL---Double Real
                    -(IEEE-754) 64 bits, (8 bytes per reading)

        DMM Power on format: SREAL
        DMM default format (when command = MFORMAT): SREAL
        """
        # self._dmm_mem_format = format
        """TODO: query cmd "MFORMAT?" can be used to questioning the current memory format"""
        if format not in ['DREAL', 'SREAL', 'DINT', 'SINT', 'ASCII', '']:
            raise ValueError('Invalid input: memory format must be DREAL, SREAL, DINT, SINT, or ASCII')
        self.driver.write(f'MFORMAT {format or ""}')
    
    def _set_output_format(self, format):
        """
        this command esignates the GPIB output format for readings sent directly to
        the controller or transferred from reading memory to the controller.
        following are the output formats that can be specified:
        ***cr lf (carriage return, line feed) being sent by the dmm to indicatethe end of the 
           transmission when the following format is specified:

            ASCII---ASCII
                    -15 bytes per reading(the cr lf (carriage return, line feed) will be sent )

        ***No cr lf (carriage return, line feed) being sent by the dmm when the following format is specified:

            SINT ---Single Integer
                    -16 bits 2's complement (2 bytes per reading)
            DINT ---Double Integer
                    -32 bits 2's complement (4 bytes per reading)
            SREAL---Single Real
                    -(IEEE-754) 32 bits, (4 bytes per reading)
            DREAL---Double Real
                    -(IEEE-754) 64 bits, (8 bytes per reading)

        DMM Power on format: ASCII
        DMM default format (when command = OFORMAT): ASCII
        """

        if format not in ['DREAL', 'SREAL', 'DINT', 'SINT', 'ASCII', '']:
            raise ValueError('Invalid input: memory format must be DREAL, SREAL, DINT, SINT, or ASCII')

        self.driver.write(f'OFORMAT {format or ""}')

    def _set_trigger_event(self, event):
        """
        this command specifies the trigger event that causes the DMM to take a reading.
        following are the trigger events that can be specified:
            AUTO ---Triggers whenever the multimeter is not busy
            EXT  ---Triggers on low-going TTL signal on the Ext Trig connector
            SGL  ---Triggers once (upon receipt of TRIG SGL) then reverts to TRIG HOLD
            HOLD ---Disables readings
            SYN  ---Triggers when the multimeter's output buffer is empty, memory is off or empty, 
                    and the controller requests data.
            LEVEL---Triggers when the input signal reaches the voltage specified 
                    by the LEVEL command on the slope specified by the SLOPE command.
            LINE ---Triggers on a zero crossing of the AC line voltage
        DMM Power on event: AUTO
        DMM default event (when command = TRIG): SGL
        """
        if event not in ['AUTO', 'EXT', 'SGL', 'HOLD', 'SYN', 'LEVEL', 'LINE']:
            raise ValueError('Invalid input: event must be AUTO, EXT, SGL, HOLD, SYNC, LEVEL, or LINE')
        self.driver.write(f'TRIG {event or ""}')
    
    def _set_arm_event(self, event, number_arms):
        """
        Defines the event that enables (arms) the trigger event (TRIG
        command). You can also use this command to perform multiple measurement
        cycles.
        """
        """
        TODO: add value check
        """
        self.driver.write(f'TRIG {event or ""},{number_arms or ""}')
    
    def _set_measurement_type(self, measurement_type = None, max_input = None, per_resolution = None):
        """
        this command specifies the measurement type.
        following are the measurement types that can be specified:
            DCV     ---DC voltage
            ACV     ---AC voltage (Parameters of this mode is set by the SETACV command)
            ACDCV   ---AC+DC voltage measurements (Parameters of this mode is set by the SETACV command)
            OHM     ---TWO-wire ohms measurements
            OHMF    ---Four-wire ohms measurements
            DCI     ---DC current
            ACI     ---AC current
            ACDCI   ---AC+DC current measurements
            FREQ    ---Frequency measurements
            PER     ---Period measurements
            DSAC    ---Direct sampling, AC coupled
            DSDC    ---Direct sampling, DC coupled
            SSAC    ---Sub-sampling, AC coupled
            SSDC    ---Sub-sampling, DC coupled

        DMM Power on event: DCV
        DMM default event (when command = FUNC): DCV
        """
        if measurement_type is not None:
            if measurement_type not in ['DCV', 'ACV', 'ACDCV', 'ACDCI', 'DCI', 'ACI', 'OHM', 'OHMF', 'FREQ', 'PER', 'DSAC', 'DSDC', 'SSAC', 'SSDC']:
                raise ValueError('Invalid input: measurement type must be DCV, ACV, ACDCI, DCI, ACI, OHM, OHMF, FREQ, PER, DSAC, DSDC, SSAC, or SSDC')
        
        # value check for max_input
        if max_input is not None:
            if measurement_type in ['DSAC', 'DSDC', 'SSAC', 'SSDC']:
                if not 0 <= max_input <= 1000:
                    raise ValueError(f'Invalid input: max_input out of range, valid input for {measurement_type} must be between 0 and 1000, and auto ranging cannot be used')
            if not (max_input == -1 or max_input == 'AUTO'):
                if not isinstance(max_input, (int, float, np.integer, np.floating)):
                    raise TypeError('Invalid input: max_input must be an integer, float or special value "AUTO"')
                match measurement_type:
                    case 'DCV' | 'ACV' | 'ACDCV':
                        if not 0 <= max_input <= 1000:
                            raise ValueError(f'Invalid input: max_input out of range, valid input for {measurement_type} must be -1, "AUTO" or between 0 and 1000')
                    case 'OHM' | 'OHMF':
                        if not 0 <= max_input <= 1.2e9:
                            raise ValueError(f'Invalid input: max_input out of range, valid input for {measurement_type} must be -1, "AUTO" or between 0 and 1.2e9')
                    case 'DCI' | 'ACI' | 'ACDCI':
                        if not 0 <= max_input <= 1.2:
                            raise ValueError(f'Invalid input: max_input out of range, valid input for {measurement_type} must be -1, "AUTO" or between 0 and 1.2')
                    # as for FREQ and PER, the range is related to FSOURCE setting, it should be checked in either main program or by the DMM itself
        # value check for per_resolution
        if per_resolution is not None:
            if not isinstance(per_resolution, (int, float, np.integer, np.floating)):
                raise TypeError('Invalid input: per_resolution must be an integer or float')
            if per_resolution < 0:
                raise ValueError('Invalid input: per_resolution must be a positive number')
            
        # send the command
        self.driver.write(f'FUNC {measurement_type or ""},{max_input or ""},{per_resolution or ""}')
    
    def _query_error(self):
        """
        The ERRSTR? command reads the least significant set bit in 
        either the error register or the auxiliary error register and then clears the bit. The 
        ERRSTR? command returns two responses separated by a comma. The first 
        response is an error number (100 Series = error register; 200 Series = auxiliary 
        error register) and the second response is a message (string) explaining the error
        """
        fetched_error = self.driver.query_ascii_values('ERRSTR?', converter='s') #decode the error message into strings
        return fetched_error
    
    def _set_nplc(self, nplc):
        """
        this command sets the number of power line cycles (NPLC) over which the measurement is made.

        DMM Power on power_line_cycles: 10
        DMM default power_line_cycles (when command = NPLC): 0 (selects minimum integration time of 500 ns)
        """
        if not isinstance(nplc, (int, float, np.integer, np.floating)):
            raise TypeError('Invalid input: nplc must be an integer or float')
        if not 0 <= nplc <= 1000:
            raise ValueError('Invalid input: nplc out of range, valid input must be between 0 and 1000')
        self.driver.write(f'NPLC {nplc or ""}')

    def _set_timer(self, time):
        """
        this command sets the time for the measurement to be made.
        TODO: set connection to the nplc
        """
        if time > 6000:
            # the smallest time is determined by sampling rate and is equal to (1 /maximum sampling rate)
            # this minimum time will be either checked by the main function or by the DMM itself
            # the largest time is 6000s
            raise ValueError('Invalid input: time out of range, valid input must be no larger than 6000')
        
        self.driver.write(f'TIMER {time or ""}')
    
    def _set_nrdgs(self, nrdgs, event):
        """
        Designates the number of readings taken per trigger and the event (sample event) that initiates each reading.
        TODO: add value check 
        """

        self.driver.write(f'NRDGS {nrdgs or ""},{event or ""}')
    
    def _set_display(self, display):
        self.driver.write(f'DISP {display or ""}')

    def _beep_once(self):
        self.driver.write('TONE')

    def _reset(self):
        self.driver.write('RESET')

    def _query_identity(self):
        return self.driver.query('ID?')
    
    def _start_self_calibration(self, acal_type = 'ALL', security_code = None):
        if acal_type not in ['ALL', 'DCV', 'AC', 'OHMS']:
            raise ValueError('Invalid input: type must be ALL, DCV, AC, or OHMS')

        self.driver.write(f'ACAL {acal_type},{security_code or ""}')

    def _start_self_test(self):
        self.driver.write('TEST')

        
class DMMinterface(DMMhwcmd):
    '''
    class for interacting with the DMM
    '''
    def __init__(self, rm, addr=12):
        super().__init__(rm, addr)
        self._apply_preset('NORM')
        self._set_trigger_event('HOLD')
        self._set_mem_format('DREAL')
        self._set_output_format('DREAL')

    def dcv_measurement(self, max_input = 'AUTO', per_resolution = None, nplc = 1, nrdgs = 1):
        # self.driver.write(f'DCV {max_input} {per_resolution}') # 100nV resolution DCV 1 10nv resolution
        # self._set_measurement_type('DCV', max_input, per_resolution)
        self._set_measurement_type(measurement_type = 'DCV', max_input = max_input, per_resolution = per_resolution)
        # self.driver.write(f'NPLC {nplc}')
        self._set_nplc(nplc)
        # self.driver.write('TIMER 200E-3') # must be smaller than nplc
        # self._set_timer(200E-3)
        # self.driver.write(f'nrdgs {nrdgs},TIMER')
        self._set_nrdgs(nrdgs, 'AUTO')

        self._set_arm_event('SYN', None)
        # self.driver.write('TARM SYN')
        self._set_trigger_event('SYN')
        # self.driver.write('TRIG SYN')
        

        message = ''
        # Sample event triggered here
        message = (self.driver.read_bytes((nrdgs*8)))
        results = []
        for i in range (int(0), (nrdgs*8), int(8)):
            eighthbyte = int(i+8)
            reading = message[i:eighthbyte]
            result = unpack('>d', reading)[0] # unpack the byte in big endian format
            result = float(result)
            results.append(result)
            # print(results) # test line

        self._set_trigger_event('HOLD') # set the trigger event back to hold
        return results
    
    def query_error_list(self):
        # dmm can query error messages
        error_list = []
        while True:
            fetched_error = self._query_error()
            # fetched_error = self.driver.query_ascii_values('ERRSTR?', converter='s') #decode the error message into strings
            # print("  error_query: code:", fetched_error[0], " message:", fetched_error[1])
            if (fetched_error[0] == '0'):
                break
            error_list.append(fetched_error)
        return error_list
    
    def check_errors(self):
        error_list = self.query_error_list()
        if error_list:
            raise HardwareError(message = f"Error from DMM: {error_list}", errors = error_list, module = "DMM")
        
    def perform_self_calibration(self, acal_type = 'DCV', security_code = None):
        TIMEOUTS = {
            'DCV': 300000,  # 5 minutes given about 2 min for normal completion
            'AC': 300000,   # 5 minutes given about 2 min for normal completion
            'OHMS': 900000, # 15 minutes given about 10 mins for normal completion
            'ALL': 1200000   # 20 minutes given about 11 mins for normal completion
        }

        if acal_type not in TIMEOUTS:
            raise ValueError("Invalid input: type must be 'DCV', 'AC', 'OHMS', or 'ALL'")

        original_timeout = self.driver.timeout
        try:
            self.driver.timeout = TIMEOUTS[acal_type]
            self._start_self_calibration(acal_type, security_code)
            self.check_errors()
            # ##### test code #####
            # raise HardwareError(
            #     message="DMM self test failed. Please refer to the manual for troubleshooting.",
            #     errors=self.query_error_list(),
            #     module="DMM"
            # )
        except Exception as e:
            if isinstance(e, HardwareError):
                raise HardwareError(
                    message=(
                        f"DMM self calibration failed for type '{acal_type}'. "
                        "Please refer to the manual for troubleshooting."
                    ),
                    errors=getattr(e, 'errors', None),
                    module="DMM"
                )
            else:
                raise e
        finally:
            self.driver.timeout = original_timeout
    
    def perform_self_test(self):
        """about 1-3 mins for a pass event"""
        TIMEOUT_SELF_TEST = 600000
        original_timeout = self.driver.timeout
        try:
            self.driver.timeout = TIMEOUT_SELF_TEST
            self._start_self_test()
            self.check_errors()

        except Exception as e:
            raise HardwareError(
                message="DMM self test failed. Please refer to the manual for troubleshooting.",
                errors=getattr(e, 'errors', None),
                module="DMM"
            )
        finally:
            self.driver.timeout = original_timeout


class SWMhwcmd:
    """SWM is acronym for Switch Matrix, originally from HP 4062C semiconductor parametric test system manual vol.1"""
    def __init__(self, rm, addr=22):
        self.rm = rm #reference to Resource manager
        self.driver = self.rm.open_resource(f'GPIB0::{addr}::INSTR')
        self.driver.timeout = 20000
        self.driver.write_termination = '\n'
        self.driver.read_termination = None
        # 48 pins and 11 ports, 7th port is ground
        # 1-24 pins correspond to 1-24 on the probe card, the rest 24 pins are labeled with letters on the probe card
        # hard coded valid ports and pins
        self.valid_ports = [1,2,3,4,5,6,7,8,9,10,11]
        self.valid_pins = list(range(1, 49))


    def _connect(self, port, pin):
        if port not in self.valid_ports:
            raise ValueError(f"Invalid port number: {port}")
        if pin not in self.valid_pins:
            raise ValueError(f"Invalid pin number: {pin}")
        # """Connects a given port to a given pin using their names."""
        # port_num = self.port_map.get(port)
        # pin_num = self.pin_map.get(pin)
        # cmd = 'PC{}ON{:02d}'.format(port,pin) 
        cmd = f'PC{port}ON{pin:02d}' # generate command string for matrix
        self.driver.write(cmd)
        #  print('matrix {} connected to {}'.format(port,pin)) # uncomment to debug

    def _clear_matrix(self):
        cmd = "CL"
        self.driver.write(cmd)

    def _query_identity(self):
        return self.driver.query('ID')

    
    """
    NOTE: the following two functions are not used in the current version of the code
        Starting relay test without installing an test adapter will directly cause failure and reboot of the SWM controller will be required
    """
    # def _start_relay_test(self):
    #     cmd = "TS"
    #     self.driver.write(cmd)

    # def _get_relay_test_result(self):
    #     cmd = "TR"
    #     result = self.driver.query(cmd)
    #     return result



class SWMinterface(SWMhwcmd):
    def __init__(self, rm, addr=22):
        super().__init__(rm, addr)
        self.pin_aliases = {}
        self.port_aliases = {}
    # aliases_dict = {
    #     "pins": {'u1':17,'u2':14,'d2':23,'d1':20,'d4':5,'u4':8,'d3':2,'u3':11},
    #     "ports": {'smu1':1,'smu2':2,'smu3':3,'smu4':4,'smu5':5,'gnd':7,'dmm_hi':8,'dmm_lo':9}
    # }
    def set_aliases(self, aliases_dict):
        """
        aliases_dict: dictionary with keys 'pins' and 'ports' and values as dictionaries with pin/port aliases as keys and pin/port numbers as values
        """
        for pin_alias, pin_num in aliases_dict['pins'].items():
            if pin_num not in self.valid_pins:
                raise ValueError(f"Invalid pin number: {pin_num} for pin alias: {pin_alias}")
        for port_alias, port_num in aliases_dict['ports'].items():
            if port_num not in self.valid_ports:
                raise ValueError(f"Invalid port number: {port_num} for port alias: {port_alias}")
        self.pin_aliases = aliases_dict['pins']
        self.port_aliases = aliases_dict['ports']
    
    def connect_ports_pins(self, pin_map):
        """
        pin_map: dictionary with pin aliases as keys and port aliases as values
        """
        self._clear_matrix()
        for pin, port in pin_map.items():
            if pin_map[pin] != None:
                if pin not in self.pin_aliases:
                    raise ValueError(f"Invalid pin alias: {pin}")
                if port not in self.port_aliases:
                    raise ValueError(f"Invalid port alias: {port}")
                self._connect(self.port_aliases[port], self.pin_aliases[pin])
    
    def check_status(self):
        """
        Checks only the serious mailfunction of the switch matrix
        """
        status = self.driver.read_stb()
        if (status & 0b01000000) and (not (status & 0b00000001)):
            error_list = []
            if status & 0b00000010:
                error_list.append("Syntax error")
            if status & 0b00010000:
                error_list.append("No pin board installed or pin board is not responding")
            raise HardwareError(message = "Switch matrix error: switch matrix is in error state and cannot proceed with the operation", errors = [], module = "SWM")
        elif status & 0b10000000:
            raise HardwareError(message = "Switch matrix faliure: switch matrix is in failure state and requires reboot", errors = [], module = "SWM")
        else:
            return status
class PBhwcmd:
    '''
    class for interacting with the probebench
    '''

    def __init__(self, rm, addr=1):
        self.rm = rm #reference to Resource manager
        # print self.rm.list_resources()
        self.driver = self.rm.open_resource(f'GPIB0::{addr}::INSTR')

        self.driver.timeout = 20000
        self.driver.write_termination = None
        self.driver.read_termination = '\r\n'
        # through stress testing found that the probebench can crash a 0.1 sec delay is sufficient to prevent the crashing
        self.delay_time = 0.1 # time in seconds to delay new writes to prevent the interface crashing
    
    def __convert_to_int(self, value):
        if value is None:
            return 2
        return int(value)
    
    def delay_decorator(func):
        @functools.wraps(func) # this will keep the original function name and docstring
        def wrapper(self, *args, **kwargs):
            time.sleep(self.delay_time)
            result = func(self, *args, **kwargs)
            time.sleep(self.delay_time)
            return result
            # print("triggered")
            # return func(self, *args, **kwargs)
        return wrapper

    @delay_decorator
    def _reset_chuck(self):
        """
        command not included in the manual preserved within the lab, 
        but it does do the job of reinitialise the chuck but it moves so fast that it might not be safe
        plus it is kinda redundant since the chuck can be directly set to load position and then set to home position
        NOT RECOMMENDED TO USE THIS COMMAND
        """
        cmd = '33'
        pbresp = self.driver.write(cmd)
        return pbresp

    @delay_decorator
    def _move_chuck_load(self):
        """
        Load resets the alignment
        """
        cmd = '3A'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp

    # @delay_decorator
    # def _query_chuck_position(self):
    #     cmd = '31'
    #     pos = self.driver.query(cmd, delay = self.delay_time)#seems need to allow a delay before read or probebench crashes and requires reset
    #     keys = ['status','x','y','z','space','command']
    #     positions = dict(zip(keys,pos.split(' ')))
    #     positions['x'] = float(positions['x'])
    #     positions['y'] = float(positions['y'])
    #     positions['z'] = float(positions['z'])
    #     positions['status'] = int(positions['status'])
    #     return positions

    @delay_decorator
    def _query_chuck_position(self):
        cmd = '31'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp

    # def move(self,x,y):
    #     self.delay()
    #     #implement relative and absolute motion
    #     cmd_move = '34 {} {}'.format(x,y)
    #     self.driver.write(cmd_move)

    @delay_decorator
    def _move_chuckXY_micron(self, dx=0.0, dy=0.0, posref="H"):
        if not isinstance(dx, (int, float, np.integer, np.floating)):
            raise ValueError(f"dx must be either int, float, or NumPy numeric type, but received type {type(dx).__name__} with value {dx}")
        if not isinstance(dy, (int, float, np.integer, np.floating)):
            raise ValueError(f"dy must be either int, float, or NumPy numeric type, but received type {type(dy).__name__} with value {dy}")
        if posref not in ["H", "Z", "C", "R"]:
            raise ValueError("posref must be either H, Z, C, or R in string format")

        cmd = f'34 {dx} {dy} {posref}'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp

    @delay_decorator
    def _set_home_position(self):
        """This is set home position"""
        cmd = '40'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp

    @delay_decorator
    def _move_chuckZ_contact(self):
        cmd = '37'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp

    @delay_decorator
    def _move_chuckZ_separation(self):
        cmd = '39'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp

    @delay_decorator
    def _move_chuckZ_align(self):
        cmd = '38'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp
    
    @delay_decorator
    def _set_chuck_vacuum(self, vacuum_on=True):
        if not isinstance(vacuum_on, bool):
            raise ValueError("vacuum_on must be a boolean value")
        if vacuum_on:
            cmd = '05 1'
        else:
            cmd = '05 0'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp
    
    @delay_decorator
    def _set_chuck_mode(self, overtravel=None, auto_Z=None, interlock=None, edge_sensor=None):

        if not isinstance(overtravel, (bool, NoneType)):
            raise ValueError("overtravel must be a boolean value or None")
        else:
            overtravel = self.__convert_to_int(overtravel)

        if not isinstance(auto_Z, (bool, NoneType)):
            raise ValueError("auto_Z must be a boolean value or None")
        else:
            auto_Z = self.__convert_to_int(auto_Z)

        if not isinstance(interlock, (bool, NoneType)):
            raise ValueError("interlock must be a boolean value or None")
        else:
            interlock = self.__convert_to_int(interlock)

        if not isinstance(edge_sensor, (bool, NoneType)):
            raise ValueError("edge_sensor must be a boolean value or None")
        else:
            edge_sensor = self.__convert_to_int(edge_sensor)

        
        cmd = f'03 {overtravel or ""} {auto_Z or ""} {interlock or ""} {edge_sensor or ""} 2'
        pbresp = self.driver.query(cmd, delay = self.delay_time)
        return pbresp
    
    
    
class PBinterface(PBhwcmd):
    def __init__(self, rm, addr=1):
        super().__init__(rm, addr)

        # to automatically decorate all methods with error checking
        for attr_name in dir(self):
            if not attr_name.startswith('__'):  # ignore built-in methods
                attr = getattr(self, attr_name)
                if callable(attr) and hasattr(PBhwcmd, attr_name):
                    decorated_method = self.check_errors_decorator(attr)
                    setattr(self, attr_name, decorated_method)

    @staticmethod
    def check_errors_decorator(method):
        def wrapper(*args, **kwargs):
            pbresp = method(*args, **kwargs)
            if pbresp and isinstance(pbresp, str) and pbresp[0] != '0':
                raise HardwareError(message=f"Error from Probebench: {pbresp}", errors=[pbresp], module="PB")
            return pbresp
        return wrapper

    def get_chuck_position(self):
        pbresp = self._query_chuck_position()
        keys = ['status','x','y','z','space','command']
        positions = dict(zip(keys,pbresp.split(' ')))
        positions['x'] = float(positions['x'])
        positions['y'] = float(positions['y'])
        positions['z'] = float(positions['z'])
        positions['status'] = int(positions['status'])
        return positions
    
    # def check_response(self, response):
    #     """simply check by rule if the response is an error"""
    #     if response[0] != '0':
    #         pb_error = True
    #     else:
    #         pb_error = False
    #     return pb_error
    
