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

from dactyl_generation.constants import *
from dactyl_generation.openai_generation import format_message_with_few_shot_examples

load_dotenv()

MISTRAL_CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def create_message_batch(file_name: str, system_prompt: str, batch_of_examples: List[List[str]], max_tokens: int) -> Tuple[List[dict], mistralai.models.UploadFileOut]:
    """
    Creates a message batch of few shot examples to pass to Mistral API.

    Args:
        file_name: Name of file in Mistral API to save as.
        system_prompt: System prompt to pass to.
        batch_of_examples: list of lists containing examples
        max_tokens: maximum number of tokens to generate

    Returns:
        tuple: List of requests sent, UploadFileOut object
    """

    buffer = BytesIO()
    list_of_requests = list()
    for index, examples in enumerate(batch_of_examples):
        messages = format_message_with_few_shot_examples(system_prompt, examples)
        request = {
            CUSTOM_ID: f"request-{index}",
            BODY: {
                MESSAGES: messages,
                "max_tokens": max_tokens,
                TEMPERATURE: np.random.uniform(0, 1),
                TOP_P: np.random.uniform(0, 1)
            }
        }
        list_of_requests.append(request)
        buffer.write((json.dumps(request)+"\n").encode("utf-8"))
    file = File(file_name=file_name, content=buffer.getvalue())
    return list_of_requests, MISTRAL_CLIENT.files.upload(file=file, purpose="batch")


def start_batch_job(input_file: mistralai.models.UploadFileOut, model: str) -> mistralai.models.BatchJobOut:
    """
    Start batch job from input file containing prompts.

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

def create_batch_job(file_name: str, system_prompt: str, examples: List[str], few_shot_size: int, model: str,max_tokens: int) -> dict:
    """
    Creates batch job for few shot prompting given examples, system prompt, and file name to upload to.
    Args:
        file_name: name of file to upload to Mistral API.
        system_prompt: System prompt.
        examples: list of examples, should be divisible by `few_shot_size`.
        few_shot_size: number of examples per prompt, should divide `examples` evenly
        model: name of model
        max_tokens: maximum number of tokens per generation

    Returns:
        info: dictionary containing batch job info
    """

    batches = np.split(np.array(examples), len(examples)//few_shot_size)
    prompts, input_file = create_message_batch(file_name, system_prompt, batches, max_tokens)
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
        rows.append(row)
    raw_prompts = data["prompts"]
    temperatures = list()
    top_ps = list()
    prompts = list()
    custom_ids = list()
    for prompt in raw_prompts:
        prompts.append("\n\n".join([str(message[CONTENT]) for message in prompt[BODY][MESSAGES]]))
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



