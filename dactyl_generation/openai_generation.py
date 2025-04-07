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
from datetime import datetime, timezone

load_dotenv()

OPENAI_CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]  # This is the default and can be omitted
)


def create_individual_request(custom_id: Any, model: str, messages:List[dict], max_completion_tokens: int, temperature: float, top_p: float) -> dict:
    """
    Creates OpenAI REST API request for a single request

    Args:
        custom_id: Custom ID of request.
        model: name of model.
        messages: List of messages to pass.
        max_completion_tokens: Max token generation limit.
        temperature: temperature
        top_p: top-p value

    Returns:
        request: individual request formatted for OpenAI REST API.
    """
    request = {CUSTOM_ID: str(custom_id), "method": "POST", "url": "/v1/chat/completions", BODY: {
        MESSAGES: messages,
        MODEL: model,
        TEMPERATURE: temperature,
        TOP_P: top_p,
        MAX_COMPLETION_TOKENS: max_completion_tokens
    }}
    return request


def create_batch_job(messages: List[List[dict]], model: str, max_tokens: int, temperatures: List[float], top_ps: List[float]) -> dict:
    """
       Creates batch job of prompts given messages and temperatures.

       Args:
           messages: List of list of messages to pass.
           model: model name
           max_tokens: maximum token generation limit
           temperatures: temperatures
           top_ps: top-p values

       Returns:
           results: dictionary containing request information
       """
    assert(len(temperatures) == len(top_ps))
    assert(len(messages) == len(temperatures))
    json_strs = list()
    requests = list()
    for i, batch in enumerate(messages):
        request = create_individual_request(f"request-{i}", model, batch, max_tokens, temperatures[i], top_ps[i])
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
        generation[TIMESTAMP] =  str(datetime.fromtimestamp(response[BODY][CREATED],tz=timezone.utc))
        generations.append(generation)
    generations = pd.DataFrame(generations)
    requests = data[INPUT_FILE]
    prompts = list()
    temperatures = list()
    top_ps = list()
    models = list()
    custom_ids =  list()
    for request in requests:
        prompt = request[BODY][MESSAGES]
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






