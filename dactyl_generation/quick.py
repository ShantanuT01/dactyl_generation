"""
Generates texts quickly using wrapper functions to redirect to appropriate model functions.
"""
import json
from dactyl_generation import openai_generation, anthropic_generation, mistral_generation
from dactyl_generation import google_generation,  bedrock_generation, deepseek_generation
from dactyl_generation.constants import *
import numpy as np
import tempfile
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from typing import List


def generate_texts_using_batch(model: str, output_path: str, prompts_df: pd.DataFrame, max_completion_tokens: int = 512) -> None:
    """
    Generates prompts to use using batch APIs from select providers using example prompts path.
    Prompt and batch data are saved to the output_path as a JSON.

    Args:
        model: Name of model.
        output_path: output path to save prompt metadata
        prompts_df: prompts for each generation
        max_completion_tokens: max completion tokens


    Returns:
        None
    """


    messages = prompts_df[MESSAGES].to_list()
    temperatures = prompts_df[TEMPERATURE].to_list()
    top_ps = prompts_df[TOP_P].to_list()

    if model.find(CLAUDE) >= 0:
        parameters = anthropic_generation.create_batch_job(messages, model, temperatures,top_ps,max_completion_tokens=max_completion_tokens)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    elif model.find(GPT) >= 0:
        parameters = openai_generation.create_batch_job(messages, model, max_completion_tokens, temperatures, top_ps)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    elif model.find(MISTRAL) >= 0:
        file_name = next(tempfile._get_candidate_names())
        file_name = f"{file_name}.jsonl"
        parameters = mistral_generation.create_batch_job(file_name,messages, model, max_tokens=max_completion_tokens, temperatures=temperatures, top_ps=top_ps)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    else:
        raise Exception("Model type not supported")


def get_batch_job_results(file_path: str, output_path: str) -> None:
    """
    Saves batch job prompts as JSON file.

    Args:
        file_path: File path containing batch data saved from `generate_texts_using_batch_with_few_shot_prompting`.
        output_path: Output JSON path to save generations.

    Returns:
        None
    """
    with open(file_path) as file:
        data = json.load(file)
    api_call = data[API_CALL]
    if api_call == ANTHROPIC:
        df = anthropic_generation.get_batch_job_output(file_path)
    elif api_call == MISTRAL:
        df = mistral_generation.get_batch_job_output(file_path)
    elif api_call == OPENAI:
        df = openai_generation.get_batch_job_output(file_path)
    else:
        raise Exception(f"API call {api_call} not supported")
    df.to_json(output_path,index=False, orient='records', indent=4)


def generate_texts_streaming(model: str, prompts_df: pd.DataFrame, output_path: str, max_completion_tokens: int =512, category: str ="", wait_after_every:int =20, sleep_time: int =30) -> None:
    """
    This function generates examples from an API live, no batching. If `example_prompts_path` is given, the function will use all prompts in the JSON file.
    Otherwise, it will generate random few shot examples.
    Outputs are saved as JSON.

    Args:
        model: name of model
        prompts_df: dataframe containing prompts
        output_path: output path to save JSON file
        max_completion_tokens: maximum number of tokens per generation
        category: categorical column
        wait_after_every: Pauses generation after a certain amount of requests
        sleep_time: Sleeps for a certain amount of time in seconds


    Returns:
        None
    """
    rows = list()

    messages = prompts_df[MESSAGES].to_list()
    temperatures = prompts_df[TEMPERATURE].to_list()
    top_ps = prompts_df[TOP_P].to_list()
    for index in tqdm(range(len(prompts_df))):

        message_batch = messages[index]

        temperature = temperatures[index]
        top_p = top_ps[index]
        row = dict()
        row[PROMPT] = message_batch
        row[TEMPERATURE] = temperature
        row[TOP_P] = top_p
        row[MODEL] = model
        row[TARGET] = 1
        row["category"] = category
        if model.find(BEDROCK) >= 0:
            text = bedrock_generation.prompt(message_batch, model, temperature, top_p, max_completion_tokens=max_completion_tokens)
        elif model.find(DEEPSEEK) >= 0:
            text = deepseek_generation.prompt(message_batch, model, temperature, top_p, max_completion_tokens=max_completion_tokens)[0]
        elif model.find(GEMINI) >= 0:
            text = google_generation.prompt(message_batch,model, temperature, top_p, max_completion_tokens)
        else:
            raise Exception("Model type not supported")
        row[TEXT] = text
        row[TIMESTAMP] = str(datetime.now(timezone.utc))
        rows.append(row)
        pd.DataFrame(rows).to_json(output_path, orient="records", indent=4, index=False)
        if (index % wait_after_every == 0) and (index > 0):
            time.sleep(sleep_time)

    pd.DataFrame(rows).to_json(output_path, orient="records", indent=4, index=False)


