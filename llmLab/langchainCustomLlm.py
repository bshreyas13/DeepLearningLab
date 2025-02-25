# -*- coding: utf-8 -*-
"""
Date:6/7/2024  
Description : This module contains the implementation of a custom Hugging Face Language Model (LLM) class and a function to create a conversation chain.
The CustomHuggingFaceLLM class is a custom implementation of the LLM class for Hugging Face models. It provides methods to initialize the LLM instance, generate a response for a given prompt, and retrieve the type of the LLM.
The create_chain function creates a conversation chain using the given model manager. It sets up a prompt template, initializes the CustomHuggingFaceLLM instance, and configures the conversation chain with a memory.
Note: This code assumes the presence of the langchain.llms.base, llmLab.modelManager, langchain.prompts, langchain.chains, and langchain.memory modules.

"""

from typing import Optional, List, Dict
from langchain.llms.base import LLM
from llmLab.modelManager import ModelManager
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

class CustomHuggingFaceLLM(LLM):
    """
    Custom implementation of the LLM class for Hugging Face models.
    """

    model_manager: ModelManager

    def __init__(self, model_manager: ModelManager):
        """
        Initializes the CustomHuggingFaceLLM instance.

        Args:
            model_manager (ModelManager): The model manager for the LLM.
        """
        super().__init__(model_manager=model_manager)

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the LLM.

        Returns:
            str: The type of the LLM.
        """
        return "custom_huggingface"
    
    def _call(self, input: str,  stop: Optional[List[str]] = None) -> str:
        """
        Generates a response for the given prompt.

        Args:
            prompt (str): The input prompt.
            stop (Optional[List[str]], optional): List of stop tokens to stop generation. Defaults to None.

        Returns:
            str: The generated response.
        """
        chat = [{"role": "user", "content": input}]
        response = self.model_manager.loader.generate_text(chat)
        return response

    class Config:
        arbitrary_types_allowed = True

def create_chain(model_manager: ModelManager):
    """
    Creates a conversation chain with the given model manager.

    Args:
        model_manager (ModelManager): The model manager for the conversation chain.

    Returns:
        ConversationChain: The created conversation chain.
    """
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template="{history}\nuser: {input}\nassistant:",
    )
    llm = CustomHuggingFaceLLM(model_manager=model_manager)
    ## Comment out conversation chain related coce
    ## Check need for conversation memory
    ## Replace with runnable
    ## return runnable / conversation , esesentially something than carries the chat like a chatbot
    
    memory = ConversationBufferMemory()  # or use ConversationSummaryMemory(llm=llm)
    conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)
    return conversation
