# -*- coding: utf-8 -*-
"""
Date:6/7/2024  

Description:
This module contains classes for loading models and tokenizers for different model loaders.

Classes:
- ModelLoader: An abstract base class defining the interface for model loaders.
- CodeGemmaLoader: A model loader for the CodeGemma model.
- DeepSeekLoader: A model loader for the DeepSeek model.
- CodestralLoader: A model loader for the Codestral model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaTokenizer, AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from abc import ABC, abstractmethod
from PIL import Image
import requests
from llmLab.decorator_utils import *

class ModelLoader(ABC):
    """
    An abstract base class defining the interface for model loaders.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Abstract method to load the model.
        """
        pass

    @abstractmethod
    def generate_text(self) -> None:
        """
        Abstract method to load the model.
        """
        pass

class Llama3Loader(ModelLoader):
    """
    A model loader for the CodeGemma model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config
    

    @auto_model_loader
    def load_model(self) -> None:
        """
        Loads the model for the CodeGemma model.
        """
        pass

    @generate_text_decorator
    def generate_text(self, chat: list) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (list): The chat input.

        Returns:
            str: The generated text.
        """
        pass

class CodeGemmaLoader(ModelLoader):
    """
    A model loader for the CodeGemma model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config
    
    @auto_model_loader
    def load_model(self) -> None:
        """
        Loads the model for the CodeGemma model.
        """
        pass

    @generate_text_decorator
    def generate_text(self, chat: list) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (list): The chat input.

        Returns:
            str: The generated text.
        """
        pass

class DeepSeekLoader(ModelLoader):
    """
    A model loader for the DeepSeek model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config
    
    @auto_model_loader
    def load_model(self) -> None:
        """
        Loads the model for the DeepSeek model.
        """
        pass

    @generate_text_decorator
    def generate_text(self, chat: list) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (list): The chat input.

        Returns:
            str: The generated text.
        """
        pass

class CodestralLoader(ModelLoader):
    """
    A model loader for the Codestral model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config

    @auto_model_loader
    def load_model(self) -> None:
        """
        Loads the model for the Codestral model.
        """
        pass

    @generate_text_decorator
    def generate_text(self, chat: list) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (list): The chat input.

        Returns:
            str: The generated text.
        """
        pass

class PaliGemmaLoader(ModelLoader):
    """
    A model loader for the PaliGemma model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config

    def load_model(self) -> None:
        """
        Loads the model for the PaliGemma model.
        """
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        torch.cuda.empty_cache()
        if self.quantization_config is not None:
            with torch.no_grad():
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    use_safetensors=True, quantization_config=self.quantization_config
                ).eval()
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    use_safetensors=True
            ).eval()
    
    @generate_vqa_text_decorator
    def generate_text(self, chat: list) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (list): The chat input.

        Returns:
            str: The generated text.
        """
        pass


