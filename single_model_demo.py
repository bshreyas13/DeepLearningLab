from transformers import BitsAndBytesConfig
from llmLab.utils import parse_args
from llmLab.modelManager import ModelManager
from llmLab.langchainCustomLlm import create_chain
import os
import torch
import traceback
import readline

if __name__ == "__main__":
    try:
        args: dict = parse_args()
        model_path: str = args["model_path"]
        
        ##TODO: Fix parallelism and remove this
        args['quantize'] = True
        if args['quantize']:
            quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_4bit=True)

        else:
            quantization_config: BitsAndBytesConfig = None
        args['quantization_config'] = quantization_config
        args.pop('quantize')
        
        manager: ModelManager = ModelManager(**args)
        manager.initialize()
        conversation = create_chain(manager)
        history = ""

        while True:
            chat_input: str = input("You: ")
            if chat_input.lower() == "exit":
                torch.cuda.empty_cache()
                args['logger'].info("Received exit command.Exiting chat.")
                print("Exiting chat. Received exit command.")
                break
            image: str = input("Image path(if vqa model being used): ")
            prompt_input = f"prompt:{chat_input}, image:{image}"
            response = conversation({"history": history, "input": prompt_input})
            print(f"Bot: {response['response']}")
            args['logger'].info(f"User: {chat_input}\nAI: {response['response']}")
            history = response['history'] + f"\nUser: {chat_input}\nAI: {response['response']}"

            if 'detect' in chat_input.lower():
                pass
            
    except Exception as e:
        torch.cuda.empty_cache()
        args['logger'].error(traceback.format_exc())
        raise e