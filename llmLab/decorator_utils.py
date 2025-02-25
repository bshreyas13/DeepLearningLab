
import torch
from PIL import Image
import llmLab.utils as utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def generate_text_decorator(func):
    """
    A decorator function that generates text based on the given chat input.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    """
    def wrapper(self, chat: list) -> str:
        """
        The wrapper function that applies the chat template, generates text, and decodes the output.

        Args:
            self: The instance of the class.
            chat (list): The list of chat messages.

        Returns:
            str: The generated response string.

        """
        inputs = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        response_string = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response_string
    return wrapper

def generate_vqa_text_decorator(func):
    """
    A decorator function that generates VQA text based on the given chat.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    def wrapper(self, chat: dict) -> str:
        """
        Wrapper function that generates VQA text based on the given chat.

        Args:
            self (object): The object instance.
            chat (list): The chat containing the prompt and image.

        Returns:
            str: The generated VQA text.
        """
        
        prompt, image = utils.parse_input(chat[0]['content'])
        image = Image.open(image)
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=False)
        return decoded
    return wrapper

def auto_model_loader(func):
    """
    A decorator function that loads the tokenizer using AutoTokennizer 
    and model for the given function using transformers.AutoModelForCauslaLM.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    """
    def wrapper(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            print("Warning: Could not load tokenizer from model path. Loading default tokenizer.")
            tokenizer_link = utils.get_supported_models(os.path.basename(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_link)

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
    return wrapper