from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import transformers
import torch

model_id = "google/codegemma-1.1-7b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype, 
)

# Enable gradient checkpointing
model.config.gradient_checkpointing = True

chat = [
    { "role": "user", "content": "Write a hello world program" },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)