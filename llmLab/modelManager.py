# -*- coding: utf-8 -*-
"""
Date: 06/02/2024
Author: Shreyas Bhat, Sarang Joshi
Maintainer : Shreyas Bhat, Sarang Joshi
E-mail:shreyasbhat13@gmail.com

Description:
This file contains the ModelManager class which is used to manage the loading and inference of a model.

TODO: Figure out parllelism to run 32 bit models on 16GB GPU
"""


from accelerate import infer_auto_device_map, dispatch_model
import torch
from llmLab.loaders import *
import os
import logging
import json

# Updated ModelManager class
class ModelManager:
    """
    Class to manage the loading and inference of a model.

    Args:
        model_path (str): The local path to the model.
        dtype (torch.dtype, optional): The data type of the model. Defaults to torch.float16.
        quantization_config (dict, optional): The quantization configuration for the model. Defaults to None.
    """
    def __init__(self, **kwargs) -> None:
        self.model_path: str = kwargs['model_path']
        self.dtype: torch.dtype = torch.float16
        self.quantization_config: dict = kwargs['quantization_config']
        self.loader: ModelLoader = None
        self.logger: logging.Logger = kwargs['logger']
        self.logger.info(f"Initializing ModelManager with model: {self.model_path}")

    def initialize(self) -> None:
        model: str = os.path.basename(self.model_path)
        s_models = json.load(open('llmLab/supported_models.json'))
        
        if model not in s_models.keys():
            raise ValueError(f"Unsupported model, got: {model}. Choose between {s_models.keys()}")
        
        if model == "codegemma-1.1-7b-it":
            self.loader = CodeGemmaLoader(self.model_path, self.dtype, self.quantization_config)
        elif model == "Meta-Llama-3-8B-Instruct":
            self.loader = Llama3Loader(self.model_path, self.dtype, self.quantization_config)
        elif model == "deepseek-coder-6.7b-instruct":
            self.loader = DeepSeekLoader(self.model_path, self.dtype, self.quantization_config)
        elif model == "Codestral-22B-v0.1":
            self.loader = CodestralLoader(self.model_path, self.dtype, self.quantization_config)
        elif model == "paligemma-3b-mix-448":
            self.loader = PaliGemmaLoader(self.model_path, self.dtype, self.quantization_config)
        
        self.logger.info(f"Loading tokenizer and model for model: {model}")
        self.loader.load_model()
        self.logger.info(f"Model loaded successfully")

        # self.logger.info(f"Inferring device map and dispatching model to devices")
        # self.infer_device_map()
        # self.dispatch_model()

    def infer_device_map(self) -> None:
        """
        Infers the device map for the model.
        """
        self.device_map = infer_auto_device_map(self.loader.model, max_memory={0: "15GB", 1: "15GB", 2: "15GB", 3: "15GB"})

    def dispatch_model(self) -> None:
        """
        Dispatches the model to the specified devices in the device map.
        """
        self.loader.model = dispatch_model(self.loader.model, device_map=self.device_map)


