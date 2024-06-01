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