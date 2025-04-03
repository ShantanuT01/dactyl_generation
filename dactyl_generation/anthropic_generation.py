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
from dactyl_generation.openai_generation import format_message_with_few_shot_examples
dotenv.load_dotenv()


ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key = os.environ['ANTHROPIC_API_KEY'],
)
API_HEADERS = {"x-api-key": os.environ['ANTHROPIC_API_KEY'], "anthropic-version": "2023-06-01"}
def get_message_batch(system_prompt: str, examples_batch: List[List[str]], model: str, temperatures: List[float], top_ps: List[float],max_tokens: int) -> List[Request]:
    """
    Generate a batch of requests from a system prompt.

    Args:
        system_prompt: System prompt.
        examples_batch: List of list of examples.
        model: model name
        temperatures: list of temperatures
        top_ps: list of top p values
        max_tokens: maximum tokens to generate

    Returns:
        requests: list of requests
    """
    requests = list()
    for index, example in enumerate(examples_batch):
        messages = format_message_with_few_shot_examples(system_prompt, example)
        # each individual request maps to one few shot set
        request = Request(
            custom_id=f"request-{index}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                temperature=temperatures[index],
                top_p=top_ps[index],
                system=system_prompt,
                messages=messages[1:],
                max_tokens=max_tokens
            )
        )
        requests.append(request)
    return requests




def request_message_batch(system_prompt: str, examples: List[str], examples_size: int, model: str, max_completion_tokens: int =512) -> dict:
    """
    Requests message batch to Anthropic API given a list of examples.

    Args:
        system_prompt: System prompt to pass to model
        examples: list of examples, divisible by `examples_size`
        examples_size: few-shot example size
        model: name of model
        max_completion_tokens: maximum number of tokens to generate

    Returns:
        request_data: requests sent to Anthropic API
    """

    batches = np.split(np.array(examples), len(examples)//examples_size)
    temperatures = np.random.uniform(low=0.0, high=1.0, size=len(batches))
    top_ps = np.random.uniform(low=0.0, high=1.0, size=len(batches))
    requests = get_message_batch(system_prompt, batches, model,temperatures, top_ps, max_completion_tokens)

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
        row[PROMPT] = "\n\n".join([message[CONTENT] for message in prompt[PARAMS][MESSAGES]])
        prompt_rows.append(row)
    ret = pd.DataFrame(prompt_rows)
    return generations.merge(ret, on=CUSTOM_ID, how='left')