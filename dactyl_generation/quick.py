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


def select_few_shot_examples_from_dataset(human_dataframe: pd.DataFrame, few_shot_size: int, prompt_example_path: str, number_of_generations: int):
    """
    Saves set of few-shot prompts from human dataset as JSON file.

    Args:
        human_dataframe: dataframe of human texts, where `text` column exists
        few_shot_size: few shot size per prompt
        prompt_example_path: JSON ouput path
        number_of_generations: number of prompts to generate

    Returns:
        None
    """
    examples = list()
    for _ in range(number_of_generations):
        examples.extend(human_dataframe[TEXT].sample(few_shot_size).to_list())

    pd.DataFrame({EXAMPLES: examples}).to_json(prompt_example_path, index=False, indent=4, orient="records")


def generate_texts_using_batch_with_few_shot_prompting(model: str, human_dataframe: pd.DataFrame, output_path: str, system_prompt: str, few_shot_size: int, number_of_generations: int = 200,max_completion_tokens: int = 512, example_prompts_path: str =None) -> None:
    """
    Generates prompts to use using batch APIs from select providers.
    If `example_prompts_path` is `None`, the function will randomly sample few-shot examples to use; otherwise it will generate prompts using the examples provided.
    Prompt and batch data are saved to the output_path as a JSON.

    Args:
        model: Name of model.
        human_dataframe: human dataframe where text column is `text`
        output_path: output path to save prompt metadata
        system_prompt: System prompt
        few_shot_size: few shot size.
        number_of_generations: number of generations
        max_completion_tokens: max completion tokens
        example_prompts_path: Examples prompts saved in JSON format from the `select_few_shot_examples_from_dataset` function.

    Returns:
        None
    """
    if example_prompts_path is None:
        examples = list()
        for _ in range(number_of_generations):
            examples.extend(human_dataframe[TEXT].sample(few_shot_size).to_list())
    else:
        examples = pd.read_json(example_prompts_path)[EXAMPLES].to_list()

    if model.find(CLAUDE) >= 0:
        parameters = anthropic_generation.request_message_batch(system_prompt, examples, few_shot_size, model, max_completion_tokens=max_completion_tokens)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    elif model.find(GPT) >= 0:
        parameters = openai_generation.create_batch_job(system_prompt, examples, few_shot_size, model, max_completion_tokens)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    elif model.find(MISTRAL) >= 0:
        file_name = next(tempfile._get_candidate_names())
        file_name = f"{file_name}.jsonl"
        parameters = mistral_generation.create_batch_job(file_name,system_prompt, examples, few_shot_size, model, max_tokens=max_completion_tokens)
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


def generate_texts_with_few_shot_prompting(model: str, human_dataframe: pd.DataFrame, output_path: str, system_prompt: str, few_shot_size: int, number_of_generations: int =200, max_completion_tokens: int =512, category: str ="", wait_after_every:int =20, sleep_time: int =30, example_prompts_path: str =None):
    """
    This function generates examples from an API live, no batching. If `example_prompts_path` is given, the function will use all prompts in the JSON file.
    Otherwise, it will generate random few shot examples.
    Outputs are saved as JSON.

    Args:
        model: name of model
        human_dataframe: human dataframe with `text` column to pull examples from.
        output_path: output path to save JSON file
        system_prompt: System prompt
        few_shot_size: few shot size
        number_of_generations: number of generations/prompts to make
        max_completion_tokens: maximum number of tokens per generation
        category: categorical column
        wait_after_every: Pauses generation after a certain amount of requests
        sleep_time: Sleeps for a certain amount of time in seconds
        example_prompts_path: Example prompts JSON

    Returns:
        None
    """
    rows = list()
    complete_examples = None
    iterations = number_of_generations
    if example_prompts_path:
        complete_examples = pd.read_json(example_prompts_path)[EXAMPLES].to_list()
        iterations = len(complete_examples)
    for count,_ in enumerate(tqdm(range(iterations))):
        if complete_examples:
            examples = complete_examples[count]
        else:
            examples = human_dataframe[TEXT].sample(few_shot_size).to_list()
        
        if model.find(BEDROCK) >= 0:
            max_temperature = 1
        else:
            max_temperature = 2
        temperature = np.random.uniform(0, max_temperature)
        top_p = np.random.uniform(0, 1)
        row = dict()
        row[PROMPT] = system_prompt + "\n\n".join(examples)
        row[TEMPERATURE] = temperature
        row[TOP_P] = top_p
        row[MODEL] = model
        row[TARGET] = 1
        row["category"] = category
        if model.find(BEDROCK) >= 0:
            text = bedrock_generation.prompt_with_few_shot_examples(system_prompt, examples, model, temperature, top_p, max_completion_tokens=max_completion_tokens)
        elif model.find(DEEPSEEK) >= 0:
            messages = openai_generation.format_message_with_few_shot_examples(system_prompt, examples)
            text = deepseek_generation.prompt_with_few_shot_examples(messages, model, temperature, top_p, max_completion_tokens=max_completion_tokens)[0]
        elif model.find(GEMINI) >= 0:
            prompt = "\n\n".join(examples)
            text = google_generation.prompt_with_few_shot_examples(system_prompt, prompt, model, temperature, top_p)
        else:
            raise Exception("Model type not supported")
        row[TEXT] = text
        rows.append(row)
        pd.DataFrame(rows).to_json(output_path, orient="records", indent=4, index=False)
        if (count % wait_after_every == 0) and (count > 0):
            time.sleep(sleep_time)


    pd.DataFrame(rows).to_json(output_path, orient="records", indent=4, index=False)


