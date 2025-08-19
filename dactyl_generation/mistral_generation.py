"""
Generates texts with using the Mistral Batch API.
"""
import copy

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


def create_message_batch(file_name: str, prompts_df: pd.DataFrame) -> Tuple[List[dict], mistralai.models.UploadFileOut]:
    """
   Creates batch of messages to send to Mistral API.

    Args:
        file_name: Name of file in Mistral API to save as.
        prompts_df: DataFrame containing prompts and generation parameters

    Returns:
        tuple: List of requests sent, UploadFileOut object
    """

    buffer = BytesIO()
    list_of_requests = list()
    messages = prompts_df.to_dict(orient="records")
    digits_length = int(np.log10(len(prompts_df))) + 1
    for index, message_batch in enumerate(messages):
        request = {
            CUSTOM_ID: f"request-{str(index).zfill(digits_length)}",
            BODY: message_batch
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

def create_batch_job(file_name: str, prompts_df: pd.DataFrame) -> dict:
    """
    Creates batch job for set of prompts given file name to save Mistral prompts to.
    Args:
        file_name: name of file to upload to Mistral API.
        prompts_df: DataFrame containing generation prompts and parameters.

    Returns:
        info: dictionary containing batch job info
    """
    assert(len(prompts_df[MODEL].unique()) == 1)
    model = prompts_df[MODEL].unique()[0]
    prompts, input_file = create_message_batch(file_name, prompts_df)
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
        row[TIMESTAMP] = str(datetime.fromtimestamp(response[RESPONSE][BODY][CREATED], tz=timezone.utc))
        rows.append(row)
    raw_prompts = pd.DataFrame([{**prompt[BODY], **{CUSTOM_ID: prompt[CUSTOM_ID]}} for prompt in data[PROMPTS]])
    print(raw_prompts.head())
    generations = pd.DataFrame(rows)
    return generations.merge(raw_prompts, on=CUSTOM_ID,how="left")



