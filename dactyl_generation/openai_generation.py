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

def create_individual_request(custom_id: str, message_body: dict) -> dict:
    """
    Creates OpenAI REST API request for a single request.
    Args:
        custom_id: Custom ID of request
        message_body: dictionary of a single message. This includes the messages, max_completion_token parameters etc.

    Returns:
        request: individual request formatted for OpenAI REST API.
    """
    request = {CUSTOM_ID: str(custom_id), "method": "POST", "url": "/v1/chat/completions", BODY: message_body}
    return request


def create_batch_job(prompts_df: pd.DataFrame) -> dict:
    """
       Creates batch job of prompts given messages and temperatures.

       Args:
           prompts_df: DataFrame where each row corresponds to an OpenAI API call.

       Returns:
           results: dictionary containing request information
       """
    digits_length = int(np.log10(len(prompts_df))) + 1
    json_strs = list()
    requests = list()
    records = prompts_df.to_dict("records")
    for i, record in enumerate(records):
        request = create_individual_request(f"request-{str(i).zfill(digits_length)}", record)
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
    requests = pd.DataFrame(data[INPUT_FILE])

    generations = generations.merge(requests, on=CUSTOM_ID, how='left')
    return generations






