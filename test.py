import numpy as np 

from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch


processor = AutoProcessor.from_pretrained("/vtti/projects06/451857/Data/Dump/ShreyasTest/paligemma-3b-mix-448")
model = PaliGemmaForConditionalGeneration.from_pretrained("/vtti/projects06/451857/Data/Dump/ShreyasTest/paligemma-3b-mix-448",device_map="cuda:0",revision="bfloat16",torch_dtype=torch.bfloat16).eval()

prompt = "detect mug"

url = "test.png"
image = Image.open(url)
# url = "https://huggingface.co/spaces/big-vision/paligemma/resolve/main/examples/cc_fox.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)
image = np.array(image)[...,:3]

inputs = processor(text=prompt, images=np.array(image), return_tensors="pt")
inputs = {name: tensor.cuda() for name, tensor in inputs.items()}

# Generate
generate_ids = model.generate(**inputs, max_length=2000)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)