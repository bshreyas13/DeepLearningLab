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

from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaTokenizer
import torch
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    """
    An abstract base class defining the interface for model loaders.
    """

    @abstractmethod
    def load_tokenizer(self) -> None:
        """
        Abstract method to load the tokenizer.
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Abstract method to load the model.
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

    def load_tokenizer(self) -> None:
        """
        Loads the tokenizer for the CodeGemma model.
        """
        self.tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-1.1-7b-it")

    def load_model(self) -> None:
        """
        Loads the model for the CodeGemma model.
        """
        torch.cuda.empty_cache()
        if self.quantization_config is not None:
            with torch.no_grad():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    use_safetensors=True, quantization_config=self.quantization_config
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=self.dtype,
                use_safetensors=True
            )

class DeepSeekLoader(ModelLoader):
    """
    A model loader for the DeepSeek model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config

    def load_tokenizer(self) -> None:
        """
        Loads the tokenizer for the DeepSeek model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)

    def load_model(self) -> None:
        """
        Loads the model for the DeepSeek model.
        """
        torch.cuda.empty_cache()
        if self.quantization_config is not None:
            with torch.no_grad():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                    use_safetensors=True, quantization_config=self.quantization_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=self.dtype,
                trust_remote_code=True,
                use_safetensors=True)

class CodestralLoader(ModelLoader):
    """
    A model loader for the Codestral model.
    """

    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config

    def load_tokenizer(self) -> None:
        """
        Loads the tokenizer for the Codestral model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/codestral-22B-v0.1", trust_remote_code=True)

    def load_model(self) -> None:
        """
        Loads the model for the Codestral model.
        """
        torch.cuda.empty_cache()
        if self.quantization_config is not None:
            with torch.no_grad():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                    use_safetensors=True, quantization_config=self.quantization_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=self.dtype,
                trust_remote_code=True,
                use_safetensors=True)
