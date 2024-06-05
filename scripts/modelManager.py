# -*- coding: utf-8 -*-
"""
Date: 06/02/2024
Author: Shreyas Bhat, Sarang Joshi
Maintainer : Shreyas Bhat, Sarang Joshi
E-mail:sbhat@vtti.vt.edu

Description:
This file contains the ModelManager class which is used to manage the loading and inference of a model.

Roadmap:
1. Load the tokenizer for the model.
2. Load the model from the local model path.
3. Infer the device map for the model.
4. Dispatch the model to the specified devices in the device map.
5. Generate text based on the given chat.
6. Test with deepseek : https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct
7. Find a to use the quamtized coderstral/codellamma models
8. Post issue for code-gemma
9. Test with code-gemma : https://huggingface.co/google/codegemma-1.1-7b-it
"""

from transformers import BitsAndBytesConfig
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GemmaTokenizer
from accelerate import infer_auto_device_map, dispatch_model
import torch
from transformers import BitsAndBytesConfig
import os
from abc import ABC, abstractmethod
from utils import parse_args

# Define an interface for model loaders
class ModelLoader(ABC):
    @abstractmethod
    def load_tokenizer(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

# Implement specific model loaders
class CodeGemmaLoader(ModelLoader):
    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config

    def load_tokenizer(self) -> None:
        self.tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-1.1-7b-it")

    def load_model(self) -> None:
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
    def __init__(self, model_path: str, dtype: torch.dtype, quantization_config: dict):
        self.model_path = model_path
        self.dtype = dtype
        self.quantization_config = quantization_config

    def load_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)

    def load_model(self) -> None:
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
            
# Updated ModelManager class
class ModelManager:
    """
    Class to manage the loading and inference of a model.

    Args:
        model_path (str): The local path to the model.
        dtype (torch.dtype, optional): The data type of the model. Defaults to torch.float16.
        quantization_config (dict, optional): The quantization configuration for the model. Defaults to None.
    """
    def __init__(self, model_path: str, dtype: torch.dtype = torch.float16, quantization_config: dict = None) -> None:
        self.model_path: str = model_path
        self.dtype: torch.dtype = dtype
        self.quantization_config: dict = quantization_config
        self.loader: ModelLoader = None

    def initialize(self, model: str) -> None:
        if model == "codegemma-1.1-7b-it":
            self.loader = CodeGemmaLoader(self.model_path, self.dtype, self.quantization_config)
        elif model == "deepseek-coder-6.7b-instruct":
            self.loader = DeepSeekLoader(self.model_path, self.dtype, self.quantization_config)
        else:
            raise ValueError(f"Unsupported model, got: {model}. Choose between {['codegemma-1.1-7b-it', 'deepseek-coder-6.7b-instruct']}")
        
        self.loader.load_tokenizer()
        self.loader.load_model()

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

    def generate_text(self, chat: str) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (str): The chat input.

        Returns:
            str: The generated text.
        """
        inputs = self.loader.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.loader.model.device)
        outputs = self.loader.model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=self.loader.tokenizer.eos_token_id)
        response_string = self.loader.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return outputs ,   response_string

if __name__ == "__main__":

    args: dict = parse_args()
    model_path: str = args["model_path"]
    model_name = os.path.basename(model_path)
    if args['quantize']:
        quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config: BitsAndBytesConfig = None

    manager: ModelManager = ModelManager(model_path, quantization_config=quantization_config)
    manager.initialize(model_name)
    while True:
        try:       
            chat_input: str = input("Enter your chat: ")
            args['logger'].info(f"User input, Enter your chat: {chat_input}")
            role: str = input("Enter the role for the bot: ")
            args['logger'].info(f"User input, Enter the role for the bot: {role}")
            if chat_input == "exit":
                break

            chat: list = [
                {"role": f"{role}", "content": f"{chat_input}"},
            ]
            outputs , response_str = manager.generate_text(chat)
            print(f"Response: {response_str}")
            args['logger'].info(f"Response: {response_str}")
        except KeyboardInterrupt:
            torch.cuda.empty_cache()
            print("Keyboard interrupt detected. Exiting...")
            args['logger'].info("Keyboard interrupt detected. Exiting...")
            break
