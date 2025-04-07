"""
Generates texts with using the Mistral Batch API.
"""
import mistralai.files
from mistralai import Mistral, File
from dotenv import load_dotenv
import os
from io import BytesIO
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime, timezone
from dactyl_generation.constants import *

load_dotenv()

MISTRAL_CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def create_message_batch(file_name: str, messages: List[List[dict]], max_tokens: int, temperatures: List[float], top_ps: List[float]) -> Tuple[List[dict], mistralai.models.UploadFileOut]:
    """
   Creates batch of messages to send to Mistral API.

    Args:
        file_name: Name of file in Mistral API to save as.
        messages: list of list of messages to pass.
        max_tokens: maximum number of tokens to generate
        temperatures: temperatures of each prompt
        top_ps: top-p values of each prompt

    Returns:
        tuple: List of requests sent, UploadFileOut object
    """
    assert(len(temperatures) == len(top_ps))
    assert(len(messages) == len(temperatures))
    buffer = BytesIO()
    list_of_requests = list()
    for index, message_batch in enumerate(messages):
        request = {
            CUSTOM_ID: f"request-{index}",
            BODY: {
                MESSAGES: message_batch,
                MAX_TOKENS: max_tokens,
                TEMPERATURE: temperatures[index],
                TOP_P: top_ps[index]
            }
        }
        list_of_requests.append(request)
        buffer.write((json.dumps(request)+"\n").encode("utf-8"))
    file = File(file_name=file_name, content=buffer.getvalue())
    return list_of_requests, MISTRAL_CLIENT.files.upload(file=file, purpose=BATCH)


def start_batch_job(input_file: mistralai.models.UploadFileOut, model: str) -> mistralai.models.BatchJobOut:
    """
    Start batch job from input file stored on Mistral API containing prompts.

    Args:
        input_file: input file object to create job with
        model: model name to use for generation

    Returns:
        batch_job: Batch job object
    """

    batch_job = MISTRAL_CLIENT.batch.jobs.create(
        input_files=[input_file.id],
        model=model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"}
    )
    return batch_job

def create_batch_job(file_name: str, messages: List[List[dict]], model: str,max_tokens: int,  temperatures: List[float], top_ps: List[float]) -> dict:
    """
    Creates batch job for set of prompts given file name to save Mistral prompts to.
    Args:
        file_name: name of file to upload to Mistral API.
        messages: List of list of messages to pass.
        model: name of model
        max_tokens: maximum number of tokens per generation
        temperatures: list of temperatures
        top_ps: list of top_p values

    Returns:
        info: dictionary containing batch job info
    """

    prompts, input_file = create_message_batch(file_name, messages,  max_tokens, temperatures, top_ps)
    batch_job = start_batch_job(input_file, model)
    input_file = input_file.model_dump(mode="json")
    batch_job = batch_job.model_dump(mode="json")
    return {"batch_job": batch_job, INPUT_FILE: input_file, PROMPTS: prompts, API_CALL: MISTRAL}



def get_batch_jobs():
    return MISTRAL_CLIENT.batch.jobs.list(
        metadata={"job_type": "testing"}
    )

def get_batch_job_output(file_path: str) -> pd.DataFrame:
    """
    Gets batch job results using saved metadata from a local JSON file.
    Args:
        file_path: local JSON file containing output of the `create_batch_job` function

    Returns:
        df: pandas DataFrame of generations.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    job_id = data["batch_job"]["id"]
    output_file = MISTRAL_CLIENT.batch.jobs.get(job_id=job_id).output_file
    content = MISTRAL_CLIENT.files.download(file_id=output_file).read().decode("utf-8")
    json_obj = "[" + ", ".join(content.splitlines()) + "]"
    responses = json.loads(json_obj)
    rows = list()
    for response in responses:
        row = dict()
        row[CUSTOM_ID] = response[CUSTOM_ID]
        row[TEXT] = response[RESPONSE][BODY][CHOICES][0][MESSAGE][CONTENT]
        row[MODEL] = response[RESPONSE][BODY][MODEL]
        row[TIMESTAMP] = str(datetime.fromtimestamp(response[RESPONSE][BODY][CREATED], tz=timezone.utc))
        rows.append(row)
    raw_prompts = data[PROMPTS]
    temperatures = list()
    top_ps = list()
    prompts = list()
    custom_ids = list()
    for prompt in raw_prompts:
        prompts.append(prompt[BODY][MESSAGES])
        temperatures.append(prompt[BODY][TEMPERATURE])
        top_ps.append(prompt[BODY][TOP_P])
        custom_ids.append(prompt[CUSTOM_ID])

    generations = pd.DataFrame(rows)
    ret = pd.DataFrame()
    ret[PROMPT] = prompts
    ret[TEMPERATURE] = temperatures
    ret[TOP_P] = top_ps
    ret[CUSTOM_ID] = custom_ids
    return generations.merge(ret, on=CUSTOM_ID,how="left")



