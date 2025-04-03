"""
Generates texts with using the Anthropic Batch API.
"""
import anthropic
import dotenv
import os
import numpy as np
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import json
import requests
import pandas as pd
from typing import List

from dactyl_generation.constants import *
dotenv.load_dotenv()


ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key = os.environ['ANTHROPIC_API_KEY'],
)
API_HEADERS = {"x-api-key": os.environ['ANTHROPIC_API_KEY'], "anthropic-version": "2023-06-01"}

def convert_openai_system_message_to_anthropic_system_message(openai_message: dict) -> dict:
    """
    Converts OpenAI system message to Anthropic API system message.
    Doesn't support cache control yet!
    Args:
        openai_message: dictionary containing system prompt

    Returns:
        anthropic_system_prompt: dictionary containing Anthropic API message
    """
    ret = dict()
    ret[TEXT] = openai_message[CONTENT]
    ret[TYPE] = TEXT
    return ret


def convert_anthropic_system_message_to_openai_system_message(anthropic_message: dict) -> dict:
    """
    Converts Anthropic API system message to OpenAI API system message.
    Doesn't support cache control yet!
    Args:
        openai_message: dictionary containing system prompt

    Returns:
        anthropic_system_prompt: dictionary containing Anthropic API message
    """
    ret = dict()
    ret[ROLE] = SYSTEM
    ret[CONTENT] = anthropic_message[TEXT]
    return ret

def get_message_batch(messages: List[List[dict]], model: str, temperatures: List[float], top_ps: List[float],max_tokens: int) -> List[Request]:
    """
    Generate a batch of requests from list of prompts

    Args:
        messages: prompts
        model: model name
        temperatures: list of temperatures
        top_ps: list of top p values
        max_tokens: maximum tokens to generate

    Returns:
        requests: list of requests
    """
    requests = list()
    for index, message_batch in enumerate(messages):
        system_messages = list()
        normal_messages = list()
        for message in message_batch:
            if message[ROLE] == SYSTEM:
                system_messages.append(convert_openai_system_message_to_anthropic_system_message(message))
            else:
                normal_messages.append(message)


        # each individual request maps to one few shot set
        request = Request(
            custom_id=f"request-{index}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                temperature=temperatures[index],
                top_p=top_ps[index],
                system=system_messages,
                messages=normal_messages,
                max_tokens=max_tokens
            )
        )
        requests.append(request)
    return requests


def create_batch_job(messages:List[List[dict]], model: str, temperatures:List[float],top_ps: List[float],max_completion_tokens: int =512) -> dict:
    """
    Requests message batch to Anthropic API given a list of examples.

    Args:
        messages: List of messages to pass
        model: name of model
        temperatures: list of temperatures for each individual prompt
        top_ps: list of top-p values
        max_completion_tokens: maximum number of tokens to generate

    Returns:
        request_data: requests sent to Anthropic API
    """
    assert(len(temperatures) == len(top_ps))
    assert(len(messages) == len(temperatures))

    requests = get_message_batch(messages, model,temperatures, top_ps, max_completion_tokens)
    message_batch = ANTHROPIC_CLIENT.messages.batches.create(requests=requests)

    return {
        BATCH_ID: message_batch.id,
        PROMPTS: requests,
        API_CALL: ANTHROPIC
    }



def get_batch_job_output(file_path: str) -> pd.DataFrame:
    """
    Gets batch job results using saved metadata from a local JSON file.
    Args:
        file_path: local JSON file containing output of the `request_batch_job` function

    Returns:
        df: pandas DataFrame of generations.
    """
    with open(file_path) as f:
        data = json.load(f)
    message_id = data[BATCH_ID]
    response = requests.get(f"https://api.anthropic.com/v1/messages/batches/{message_id}/results",headers=API_HEADERS)
    lines = response.text.splitlines()
    objects = list()
    for line in lines:
        objects.append(json.loads(line))
    generations = list()
    for object in objects:
        generation = dict()
        generation[CUSTOM_ID] = object[CUSTOM_ID]
        generation[TEXT] = object[RESULT][MESSAGE][CONTENT][0][TEXT]
        generation[MODEL] = object[RESULT][MESSAGE][MODEL]
        generations.append(generation)
    generations = pd.DataFrame(generations)
    prompt_rows = list()
    for prompt in data[PROMPTS]:
        row = dict()
        row[CUSTOM_ID] = prompt[CUSTOM_ID]
        row[TEMPERATURE] = prompt[PARAMS][TEMPERATURE]
        row[TOP_P] = prompt[PARAMS][TOP_P]
        system_prompts = prompt[PARAMS][SYSTEM]
        prompts = list()
        for system_prompt in system_prompts:
            prompts.append(convert_anthropic_system_message_to_openai_system_message(system_prompt))
        prompts.extend(prompt[PARAMS][MESSAGES])
        row[PROMPT] = prompts
        prompt_rows.append(row)
    ret = pd.DataFrame(prompt_rows)
    return generations.merge(ret, on=CUSTOM_ID, how='left')