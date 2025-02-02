import os
from openai import OpenAI
from dotenv import load_dotenv
from dactyl_generation.constants import *
import pandas as pd
import json
import numpy as np
from io import BytesIO

load_dotenv()

OPENAI_CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]  # This is the default and can be omitted
)


def format_message_with_few_shot_examples(system_prompt: str,  examples: list) -> list:
    """
    Formats message with examples to pass to OpenAI API.
    :param system_prompt: System prompt to pass to ChatGPT
    :param examples: String examples
    :return: list of messages
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

def create_individual_request_for_batch(custom_id, model, system_prompt, examples, max_completion_tokens):
    request = {CUSTOM_ID: str(custom_id), "method": "POST", "url": "/v1/chat/completions", BODY: {
        MESSAGES: format_message_with_few_shot_examples(system_prompt, examples),
        MODEL: model,
        TEMPERATURE: np.random.uniform(0, 2),
        TOP_P: np.random.uniform(0, 1),
        MAX_COMPLETION_TOKENS: max_completion_tokens
    }}
    return request


def create_batch_job(system_prompt, examples, few_shot_size, model, max_tokens):
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


def get_batch_job_output(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)
    batch_job = OPENAI_CLIENT.batches.retrieve(data[RESULT_FILE_ID])
    print(batch_job)
    result = OPENAI_CLIENT.files.content(batch_job.output_file_id).content
    df = pd.read_json(BytesIO(result), lines=True)
    responses = df[RESPONSE]
    generations = list()
    for response in responses:
        generation = dict()
        generation[TEXT] = response[BODY][CHOICES][0][MESSAGE][CONTENT]
        generation[CUSTOM_ID] = response[CUSTOM_ID]
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


def prompt_with_few_shot_learning(messages, model, temperature, top_p, max_completion_tokens=512,number_of_responses=1):
    """
    Prompt OpenAI model with few shot learning examples.
    :param messages: List of messages to pass in
    :param model: model name
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



if __name__ == "__main__":
    df = pd.read_csv("../datasets/tweets/release/human_training.csv")
    examples = df["text"].sample(5).to_list()
    examples_2 = df["text"].sample(5).to_list()
    system_prompt = "You are Twitter bot simulating tweets from 538's 3 Million Troll Tweets dataset. You will generate human tweets in the style of that dataset. Any events referenced in tweets, if applicable, should take place between 2015 to 2018. The tweets do not have to be factual."
    request1 = create_individual_request_for_batch("custom_id_1", "gpt-4o-2024-11-20", system_prompt, examples,  100)
    request2 = create_individual_request_for_batch("custom_id_2", "gpt-4o-2024-11-20", system_prompt, examples_2,  100)
    '''
    requests = [request1, request2]
    json_strs = list()
    import json
    for request in requests:
        json_strs.append(json.dumps(request))
    with open("../datasets/openai_batch_test.jsonl",'w+') as f:
        f.write("\n".join(json_strs))
    batch_file = OPENAI_CLIENT.files.create(
        file=open("../datasets/openai_batch_test.jsonl", "rb"),
        purpose="batch"
    )
    batch_job = OPENAI_CLIENT.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    '''
    batch_job = OPENAI_CLIENT.batches.retrieve("batch_679b94756878819096547751fe5d5ef0")

    print(batch_job)
    result_file_id = batch_job.output_file_id
    from io import BytesIO
    result = OPENAI_CLIENT.files.content(result_file_id).content
    df = pd.read_json(BytesIO(result), lines=True)
    print(df["custom_id"])
    responses = df["response"]
    for response in responses:
        print(response['body']['choices'][0]['message']['content'])






