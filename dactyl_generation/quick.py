import json
from dactyl_generation import openai_generation, anthropic_generation, mistral_generation
from dactyl_generation import google_generation,  bedrock_generation, deepseek_generation
from dactyl_generation.constants import *
import numpy as np
import tempfile
import time
import pandas as pd
from tqdm import tqdm

def generate_texts_using_batch(model, human_dataframe, output_path, system_prompt, few_shot_size, number_of_generations,max_completion_tokens):
    examples = list()
    for _ in range(number_of_generations):
        examples.extend(human_dataframe[TEXT].sample(few_shot_size).to_list())
    if model.find("claude") >= 0:
        parameters = anthropic_generation.request_message_batch(system_prompt, examples, few_shot_size, model, max_completion_tokens=max_completion_tokens)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    elif model.find("gpt") >= 0:
        parameters = openai_generation.create_batch_job(system_prompt, examples, few_shot_size, model, max_completion_tokens)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    elif model.find("mistral") >= 0:
        file_name = next(tempfile._get_candidate_names())
        file_name = f"{file_name}.jsonl"
        parameters = mistral_generation.create_batch_job(file_name,system_prompt, examples, few_shot_size, model, max_tokens=max_completion_tokens)
        with open(output_path, 'w+') as file:
            json.dump(parameters, file, indent=4)
    else:
        raise Exception("Model type not supported")


def get_batch_job_results(file_path, output_path):
    with open(file_path) as file:
        data = json.load(file)
    api_call = data[API_CALL]
    if api_call == "anthropic":
        df = anthropic_generation.get_batch_job_output(file_path)
    elif api_call == "mistral":
        df = mistral_generation.get_batch_job_output(file_path)
    elif api_call == "openai":
        df = openai_generation.get_batch_job_output(file_path)
    else:
        raise Exception(f"API call {api_call} not supported")
    df.to_json(output_path,index=False, orient='records', indent=4)


def generate_texts(model, human_dataframe, output_path, system_prompt, few_shot_size, number_of_generations, max_completion_tokens,category="", wait_after_every=20, sleep_time=30):
    rows = list()
    for count,_ in enumerate(tqdm(range(number_of_generations))):
        examples = human_dataframe[TEXT].sample(few_shot_size).to_list()
        if model.find("bedrock") >= 0:
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
        if model.find("bedrock") >= 0:
            text = bedrock_generation.prompt_with_few_shot_examples(system_prompt, examples, model, temperature, top_p, max_completion_tokens=max_completion_tokens)
        elif model.find("deepseek") >= 0:
            messages = openai_generation.format_message_with_few_shot_examples(system_prompt, examples)
            text = deepseek_generation.prompt_with_few_shot_examples(messages, model, temperature, top_p, max_completion_tokens=max_completion_tokens)[0]
        elif model.find("gemini") >= 0:
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

if __name__ == "__main__":
    system_prompt = '''You are a Twitter bot simulating tweets from 538's 3 Million Troll Tweets dataset. You will generate human tweets in the style of that dataset. Any events referenced in tweets, if applicable, should take place between 2015 to 2018. The tweets do not have to be factual. You do not have to follow proper grammar rules. Output only the text of the tweet. Output only one tweet.'''


    human_dataframe = pd.read_csv("../datasets/tweets/release/human_training.csv")
    output_path = f"../batches/requests/test_speed.json"
    #generate_texts_using_batch("claude-3-haiku-20240307", human_dataframe, output_path, system_prompt, few_shot_size=5, number_of_generations=2, max_completion_tokens=100)
    generate_texts("bedrock/us.meta.llama3-2-90b-instruct-v1:0", human_dataframe, output_path, system_prompt, few_shot_size=5, number_of_generations=5, max_completion_tokens=100)
    #get_batch_job_results("../batches/requests/openai-test.json", "../batches/outputs/openai-test.json")

