from mistralai import Mistral, File
from dotenv import load_dotenv
import os
from io import BytesIO
import json
import numpy as np
import pandas as pd

from dactyl_generation.constants import *
from dactyl_generation.openai_generation import format_message_with_few_shot_examples

load_dotenv()

MISTRAL_CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def create_message_batch(file_name, system_prompt, batch_of_examples, max_tokens):
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


def start_batch_job(input_file, model):
    batch_job = MISTRAL_CLIENT.batch.jobs.create(
        input_files=[input_file.id],
        model=model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"}
    )
    return batch_job

def create_batch_job(file_name, system_prompt, examples, few_shot_size, model,max_tokens):
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

def get_batch_job_output(file_path):
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
    for prompt in raw_prompts:
        prompts.append("\n\n".join([str(message[CONTENT]) for message in prompt[BODY][MESSAGES]]))
        temperatures.append(prompt[BODY][TEMPERATURE])
        top_ps.append(prompt[BODY][TOP_P])

    generations = pd.DataFrame(rows)
    ret = pd.DataFrame()
    ret[PROMPT] = prompts
    ret[TEMPERATURE] = temperatures
    ret[TOP_P] = top_ps
    return generations.merge(ret, on=CUSTOM_ID,how="left")



