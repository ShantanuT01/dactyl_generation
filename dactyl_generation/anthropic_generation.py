import anthropic
import dotenv
import os
import numpy as np
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import json
import requests
import pandas as pd

from dactyl_generation.constants import *
from dactyl_generation.openai_generation import format_message_with_few_shot_examples
dotenv.load_dotenv()


ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key = os.environ['ANTHROPIC_API_KEY'],
)
API_HEADERS = {"x-api-key": os.environ['ANTHROPIC_API_KEY'], "anthropic-version": "2023-06-01"}
def get_message_batch(system_prompt, examples_batch, model, temperatures, top_ps,max_tokens):
    """
    Generate a batch of messages from a system prompt.

    :param system_prompt: prompt for system
    :param examples_batch: list of individual examples
    :param model: Anthropic model name
    :param temperatures: array of temperatures
    :param top_ps: array of top-p values
    :param max_tokens: maximum number of tokens
    :return: list of requests to pass to API
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




def request_message_batch(system_prompt, examples, examples_size, model, max_completion_tokens=512):
    """
    Requests message batch to Anthropic API given a list of examples.

    :param system_prompt: System prompt to pass to model
    :param examples: list of examples, divisible by `examples_size`
    :param examples_size: few-shot example size
    :param model: name of model
    :param max_completion_tokens: maximum number of tokens to generate
    :return: requests sent to Anthropic API
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


def get_batch_job_output(file_path):
    """
    Get results from Anthropic API for a batch request.
    :param file_path: file path containing metadata for request
    :return: list of dictionaries containing results from Anthropic API
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