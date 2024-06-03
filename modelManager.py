from transformers import BitsAndBytesConfig
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GemmaTokenizer
from accelerate import infer_auto_device_map, dispatch_model
import torch
from transformers import BitsAndBytesConfig

class ModelManager:
    """
    Class to manage the loading and inference of a model.

    Args:
        local_model_path (str): The local path to the model.
        dtype (torch.dtype, optional): The data type of the model. Defaults to torch.float16.
        quantization_config (dict, optional): The quantization configuration for the model. Defaults to None.
    """

    def __init__(self, local_model_path: str, dtype: torch.dtype = torch.float16, quantization_config: dict = None) -> None:
        self.local_model_path: str = local_model_path
        self.dtype: torch.dtype = dtype
        self.quantization_config: dict = quantization_config

        self.tokenizer: GemmaTokenizer = None
        self.model: AutoModelForCausalLM = None
        self.device_map: dict = None

    def load_tokenizer(self) -> None:
        """
        Loads the tokenizer for the model.
        """
        self.tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-1.1-7b-it")

    def load_model(self) -> None:
        """
        Loads the model from the local model path.
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=self.dtype,
                                                              use_safetensors=True,
                                                              quantization_config=self.quantization_config)

    def infer_device_map(self) -> None:
        """
        Infers the device map for the model.
        """
        self.device_map = infer_auto_device_map(self.model, max_memory={0: "15GB", 1: "15GB", 2: "15GB", 3: "15GB"})

    def dispatch_model(self) -> None:
        """
        Dispatches the model to the specified devices in the device map.
        """
        self.model = dispatch_model(self.model, device_map=self.device_map)

    def generate_text(self, chat: str) -> str:
        """
        Generates text based on the given chat.

        Args:
            chat (str): The chat input.

        Returns:
            str: The generated text.
        """
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=150)
        return outputs

# Usage example
local_model_path: str = "/vtti/projects06/451857/Data/Dump/ShreyasTest/codegemma-1.1-7b-it"
quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_4bit=True)

manager: ModelManager = ModelManager(local_model_path, quantization_config=quantization_config)
manager.load_tokenizer()
manager.load_model()
manager.infer_device_map()
manager.dispatch_model()

chat: list = [
    {"role": "user", "content": "Write a hello world program"},
]

outputs: str = manager.generate_text(chat)
print(outputs[0]["generated_text"])