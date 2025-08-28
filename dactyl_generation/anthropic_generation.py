"""
Generates texts with using the Anthropic Batch API.
"""
import copy

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
from datetime import datetime, timezone
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

def get_message_batch(prompts_df: pd.DataFrame) -> List[Request]:
    """
    Generate a batch of requests from list of prompts

    Args:
        prompts_df: DataFrame where each row is an API call to the Anthropic API.

    Returns:
        requests: list of requests
    """
    requests = list()
    calls = prompts_df.to_dict(orient="records")
    digits_length = int(np.log10(len(calls))) + 1
    for i, call in enumerate(calls):
        system_messages = list()
        normal_messages = list()
        for message in call[PROMPT]:
            if message[ROLE] == SYSTEM:
                system_messages.append(convert_openai_system_message_to_anthropic_system_message(message))
            else:
                normal_messages.append(message)

        call[SYSTEM] = system_messages
        call[MESSAGES] = normal_messages
        message_parameters = copy.copy(call)
        del message_parameters[PROMPT]
        # each individual request maps to one few shot set
        request = Request(
            custom_id=f"request-{str(i).zfill(digits_length)}",
            params=MessageCreateParamsNonStreaming(
                **message_parameters
            )
        )
        requests.append(request)
    return requests


def create_batch_job(prompts_df: pd.DataFrame) -> dict:
    """
    Requests message batch to Anthropic API given a list of examples.

    Args:
        prompts_df: Dataframe containing prompts to run.

    Returns:
        request_data: requests sent to Anthropic API
    """


    requests = get_message_batch(prompts_df)
    custom_ids = [request[CUSTOM_ID] for request in requests]
    message_batch = ANTHROPIC_CLIENT.messages.batches.create(requests=requests)
    prompts_df[CUSTOM_ID] = custom_ids
    return {
        BATCH_ID: message_batch.id,
        PROMPTS: prompts_df.to_dict(orient='records'),
        API_CALL: ANTHROPIC,
        TIMESTAMP:  str(datetime.now(timezone.utc))
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
        generations.append(generation)
    generations = pd.DataFrame(generations)
    generations[TIMESTAMP] = data[TIMESTAMP]
    prompt_rows = pd.DataFrame(data[PROMPTS])
    ret = pd.DataFrame(prompt_rows)
    return generations.merge(ret, on=CUSTOM_ID, how='left')