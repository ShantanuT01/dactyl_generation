"""
Generates texts with using the OpenAI Batch API.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from dactyl_generation.constants import *
import pandas as pd
import json
import numpy as np
from io import BytesIO
from typing import List, Any

load_dotenv()

OPENAI_CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]  # This is the default and can be omitted
)


def format_message_with_few_shot_examples(system_prompt: str,  examples: List[str]) -> List[dict]:
    """
    Formats few-shot example as message to pass to OpenAI API.

    Args:
        system_prompt: System prompt to pass to OpenAI API
        examples: list of examples

    Returns:
        messages: list of messages
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

def create_individual_request_for_batch(custom_id: Any, model: str, system_prompt: str, examples: List[str], max_completion_tokens: int) -> dict:
    """
    Creates OpenAI REST API request for a single few-shot example for batching.

    Args:
        custom_id: Custom ID of request.
        model: name of model.
        system_prompt: System prompt to pass.
        examples: List of examples.
        max_completion_tokens: Max token generation limit.

    Returns:
        request: individual request formatted for OpenAI REST API.
    """
    request = {CUSTOM_ID: str(custom_id), "method": "POST", "url": "/v1/chat/completions", BODY: {
        MESSAGES: format_message_with_few_shot_examples(system_prompt, examples),
        MODEL: model,
        TEMPERATURE: np.random.uniform(0, 2),
        TOP_P: np.random.uniform(0, 1),
        MAX_COMPLETION_TOKENS: max_completion_tokens
    }}
    return request

def create_batch_job_with_different_system_prompts(system_prompts: List[str], batches: List[List[str]], model: str, max_tokens: int) -> dict:
    """
       Creates batch job of prompts of few shot examples with different system prompts.

       Args:
           system_prompts: System prompt to pass.
           batches: List of list of examples.
           model: model name
           max_tokens: maximum token generation limit

       Returns:
           results: dictionary containing request information
       """

    json_strs = list()
    requests = list()
    for i, batch in enumerate(batches):
        request = create_individual_request_for_batch(f"request-{i}", model, system_prompts[i], batch, max_tokens)
        requests.append(request)
        json_strs.append(json.dumps(request))
    buffer = BytesIO(("\n".join(json_strs)).encode("utf-8"))
    # with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as fp:
    #    fp.write("\n".join(json_strs))
    #    temp_filename = fp.name

    batch_file = OPENAI_CLIENT.files.create(
        file=buffer,
        purpose="batch"
    )
    #  os.remove(temp_filename)

    batch_job = OPENAI_CLIENT.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    result_file_id = batch_job.id

    return {
        RESULT_FILE_ID: result_file_id,
        INPUT_FILE: requests,
        API_CALL: OPENAI
    }


def create_batch_job(system_prompt: str, examples: List[str], few_shot_size: int, model: str, max_tokens: int) -> dict:
    """
    Creates batch job of prompts of few shot examples.

    Args:
        system_prompt: System prompt to pass.
        examples: List of examples.
        few_shot_size: Number of examples per prompt. `len(examples)` should be divisible by `few_shot_size`.
        model: model name
        max_tokens: maximum token generation limit

    Returns:
        results: dictionary containing request information
    """
    batches = np.split(np.array(examples), len(examples)//few_shot_size)
    json_strs = list()
    requests = list()
    for i, batch in enumerate(batches):
        request = create_individual_request_for_batch(f"request-{i}", model, system_prompt, examples, max_tokens)
        requests.append(request)
        json_strs.append(json.dumps(request))
    buffer = BytesIO(("\n".join(json_strs)).encode("utf-8"))
   # with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as fp:
    #    fp.write("\n".join(json_strs))
    #    temp_filename = fp.name

    batch_file = OPENAI_CLIENT.files.create(
        file=buffer,
        purpose="batch"
    )
  #  os.remove(temp_filename)

    batch_job = OPENAI_CLIENT.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    result_file_id = batch_job.id

    return {
        RESULT_FILE_ID: result_file_id,
        INPUT_FILE: requests,
        API_CALL: OPENAI
    }


def get_batch_job_output(file_path: str) -> pd.DataFrame:
    """
    Gets batch job results using saved metadata from a local JSON file.
    Args:
        file_path: local JSON file containing output of the `create_batch_job` function

    Returns:
        df: pandas DataFrame of generations.
    """
    with open(file_path,'r') as f:
        data = json.load(f)
    batch_job = OPENAI_CLIENT.batches.retrieve(data[RESULT_FILE_ID])
    result = OPENAI_CLIENT.files.content(batch_job.output_file_id).content
    df = pd.read_json(BytesIO(result), lines=True)
    responses = df[RESPONSE]
    custom_ids = df[CUSTOM_ID]
    generations = list()
    for response, custom_id in zip(responses, custom_ids):
        generation = dict()
        generation[TEXT] = response[BODY][CHOICES][0][MESSAGE][CONTENT]
        generation[CUSTOM_ID] = custom_id
        generations.append(generation)
    generations = pd.DataFrame(generations)
    requests = data["input_file"]
    prompts = list()
    temperatures = list()
    top_ps = list()
    models = list()
    custom_ids =  list()
    for request in requests:
        prompt = "\n\n".join([message[CONTENT] for message in request[BODY][MESSAGES]])
        prompts.append(prompt)
        temperatures.append(request[BODY][TEMPERATURE])
        top_ps.append(request[BODY][TOP_P])
        models.append(request[BODY][MODEL])
        custom_ids.append(request[CUSTOM_ID])

    ret = pd.DataFrame()
    ret[PROMPT] = prompts
    ret[MODEL] = models
    ret[TEMPERATURE] = temperatures
    ret[TOP_P] = top_ps
    ret[CUSTOM_ID] = custom_ids
    generations = generations.merge(ret, on=CUSTOM_ID, how='left')
    return generations


def prompt_with_few_shot_learning(messages: List[dict], model: str, temperature: float, top_p: float, max_completion_tokens:int = 512,number_of_responses: int = 1) -> List[str]:
    """
    Get output from single few-shot prompt (live) request from OpenAI API.
    Args:
        messages: List of messages to pass in.
        model: Model name.
        temperature: temperature value, from 0 to 2.
        top_p: top-p parameter, from 0 to 1.
        max_completion_tokens: maximum token generation
        number_of_responses: max responses

    Returns:
        responses: List of generated responses
    """
    """
    Prompt OpenAI model with few shot learning examples.
    :param messages: 
    :param model: 
    :param temperature: temperature parameter
    :param top_p: top p parameter
    :param max_completion_tokens: maximum number of tokens to generate
    :param number_of_responses: max number of responses
    :return:
    """

    api_response = OPENAI_CLIENT.chat.completions.create(
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




