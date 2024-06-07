# -*- coding: utf-8 -*-
"""
Date: 06/02/2024
Author: Shreyas Bhat, Sarang Joshi
Maintainer : Shreyas Bhat, Sarang Joshi
E-mail:shreyasbhat13@gmail.com

Description:
This file contains the ModelManager class which is used to manage the loading and inference of a model.

"""


from accelerate import infer_auto_device_map, dispatch_model
import torch
from llmLab.loaders import ModelLoader, CodeGemmaLoader, DeepSeekLoader, CodestralLoader
import os
import logging

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
        if model == "codegemma-1.1-7b-it":
            self.loader = CodeGemmaLoader(self.model_path, self.dtype, self.quantization_config)
        elif model == "deepseek-coder-6.7b-instruct":
            self.loader = DeepSeekLoader(self.model_path, self.dtype, self.quantization_config)
        elif model == "Codestral-22B-v0.1":
            self.loader = CodestralLoader(self.model_path, self.dtype, self.quantization_config)
        else:
            raise ValueError(f"Unsupported model, got: {model}. Choose between {['codegemma-1.1-7b-it', 'deepseek-coder-6.7b-instruct', 'Codestral-22B-v0.1']}")
        self.logger.info(f"Loading tokenizer and model for model: {model}")
        self.loader.load_tokenizer()
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

    def generate_text(self, chat: list) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (list): The chat input.

        Returns:
            str: The generated text.
        """
        inputs = self.loader.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.loader.model.device)
        outputs = self.loader.model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=self.loader.tokenizer.eos_token_id)
        response_string = self.loader.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response_string



