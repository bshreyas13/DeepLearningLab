# -*- coding: utf-8 -*-
"""
Date:3/15/2024  
Author: Shreyas Bhat
Maintainer : Shreyas Bhat
E-mail:sbhat@vtti.vt.edu
"""
import os
import subprocess as sp
from typing import Dict, List, Union
import logging
from datetime import datetime  
import pytz
from typing import Tuple
import subprocess
from argparse import ArgumentParser
import json

def is_nested_directory(path: str) -> bool:
        """
        Check if a given path is a nested directory.

        Args:
                path (str): The path to check.

        Returns:
                bool: True if the path is a nested directory, False otherwise.

        Raises:
                ValueError: If the path does not exist.
        """
        if not os.path.exists(path):
                raise ValueError(f"Invalid Path Error, No such file or directory! {path}")
        
        if not os.path.isdir(path):
                return False
        
        # Get a list of all entries in the directory
        entries = os.listdir(path)
        
        # Check if any of the entries are directories
        for entry in entries:
                if os.path.isdir(os.path.join(path, entry)):
                        return True
        
        return False


def get_available_gpus(leave_unmasked: int = 1, memory_threshold: int = 1024) -> Tuple[Dict[str, int], List[int]]:
        """
        Get the available GPUs in the system based on memory availability.

        Args:
                leave_unmasked (int): The minimum number of GPUs that should remain unmasked.
                memory_threshold (int): The minimum memory threshold (in MB) for a GPU to be considered available.

        Returns:
                tuple: A tuple containing two elements:
                        - A dictionary mapping the available GPUs to their corresponding free memory values.
                        - A list of GPU device indices for the available GPUs.

        Raises:
                ValueError: If the number of usable GPUs found is less than the specified `leave_unmasked` value.
        """
        ACCEPTABLE_AVAILABLE_MEMORY: int = memory_threshold
        COMMAND_1: str = "nvidia-smi --query-gpu=memory.free --format=csv"
        COMMAND_2: str = "nvidia-smi --query-gpu=gpu_name --format=csv"
        try:
                _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
                memory_free_info: List[str] = _output_to_list(sp.check_output(COMMAND_1.split()))[1:]
                name_info: List[str] = _output_to_list(sp.check_output(COMMAND_2.split()))[1:]
                memory_free_values: List[int] = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
                available_gpus: List[str] = [name_info[i] + f'_{i}' for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]
                gpu_devs: List[int] = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]
                if len(available_gpus) < leave_unmasked:
                        raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
                return dict(zip(available_gpus, memory_free_values)), gpu_devs
        except Exception as e:
                print('"nvidia-smi" is probably not installed. GPUs are not masked', e)
                return {}, []

def init_logger(log_path: str) -> logging.Logger:
        """
        Initializes a logger object.

        Args:
                log_path (str): The path to the directory where the log file will be saved.

        Returns:
                logging.Logger: The initialized logger object.
        """
        log_name: str = f"{datetime.now(pytz.timezone('US/Eastern')).strftime('%m%d%Y')}.log"  
        filename: str = f'{os.path.join(log_path, log_name)}'
        os.makedirs(log_path, exist_ok=True)
        logger: logging.Logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler: logging.FileHandler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

def parse_args():
    """
    Parse command line arguments for 3D detection.

    Returns:
        init_args (dict): Dictionary containing the initial arguments.
        call_args (dict): Dictionary containing the parsed command line arguments.
        added_args (dict): Dictionary containing additional arguments.
    """
    parser = ArgumentParser()
    
    # Load arguments from JSON file
    with open('args.json') as f:
        args = json.load(f)
    # Add arguments to parser
    print("START: Adding arguments to parser")
    for arg in args:
        print(f"{arg['name']} , {arg['options']}")
        if 'type' in arg['options']:
            arg['options']['type'] = get_type_constructor(arg['options']['type'])
        parser.add_argument(arg['name'], **arg['options'])
    
    call_args = vars(parser.parse_args())
    logger = init_logger(log_path=call_args['log_path'])
    logger.info(f'Input command line arguments: {call_args}')
    call_args['logger'] = logger

    return call_args

def get_type_constructor(input_type: str):
    # List of known type-casting functions
    supported_type_constructors = {
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "complex": complex,
                "bytes": bytes,
                "bytearray": bytearray,
                "list": list,
                "tuple": tuple,
                "set": set,
                "dict": dict,
                "frozenset": frozenset
                }
    # Get all type constructors from builtins and check about supported type constructors
    #all_builtins = [name for name in dir(builtins) if name[0].islower() and callable(getattr(builtins, name))]
    if input_type in list(supported_type_constructors.keys()):
        return supported_type_constructors[input_type]
    else:
        print(f"{input_type} is not a supported type constructor. Supported type constructors are: {list(supported_type_constructors.keys())}")
        return None
    
def assert_with_log(condition, message, logger):
    try:
        assert condition, message
    except AssertionError as e:
        logger.error(f"Assertion failed: {message}")
        raise AssertionError(message) from None  # Optionally re-raise the AssertionError to halt the program