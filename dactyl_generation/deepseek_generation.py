"""
Generates texts with DeepSeek models using the DeepInfra API.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from dactyl_generation.constants import *
import pandas as pd
from typing import List
load_dotenv()

DEEPSEEK_CLIENT = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai"
)

def format_message_with_few_shot_examples(system_prompt: str,  examples: List[str]) -> list:
    """
    Formats message with examples to pass to OpenAI API which is the same format for DeepInfra's API.

    Args:
        system_prompt: System prompt to pass to DeepSeek.
        examples: String examples

    Returns:
        messages: list of messages to pass to API.
    """


    messages = list()
    messages.append(
        {
            ROLE: SYSTEM,
            CONTENT: system_prompt
        }
    )
    for example in examples:
        message = dict()
        message[ROLE] = USER
        message[CONTENT] = example
        messages.append(message)
    return messages


def prompt_with_few_shot_examples(messages: List[dict], model: str, temperature:float , top_p:float, max_completion_tokens:int=512,number_of_responses:int=1) -> list:
    """
    Pass a single list of messages to DeepSeek to generate text.

    Args:
        messages: List of messages to pass in.
        model: model name.
        temperature: temperature, value from 0 to 2.
        top_p: top-p value from 0 to 1.
        max_completion_tokens: maximum number of completion tokens
        number_of_responses: maximum number of responses.

    Returns:
        responses: List of responses.
    """


    api_response = DEEPSEEK_CLIENT.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        n=number_of_responses
    )
    responses = list()
    for response in api_response.choices:
        responses.append(response.message.content.strip())
    return responses



