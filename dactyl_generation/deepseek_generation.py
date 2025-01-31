import os
from openai import OpenAI
from dotenv import load_dotenv
from dactyl_generation.constants import *
import pandas as pd

load_dotenv()

DEEPSEEK_CLIENT = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai"
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


def prompt_with_few_shot_examples(messages, model, temperature, top_p, max_completion_tokens=512,number_of_responses=1):
    """

    :param messages: List of messages to pass in
    :param model: model name
    :param temperature:
    :param top_p:
    :param max_completion_tokens:
    :param number_of_responses:
    :return:
    """

    api_response = DEEPSEEK_CLIENT.chat.completions.create(
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
    messages = format_message_with_few_shot_examples(
        "You are Twitter bot simulating tweets from 538's 3 Million Troll Tweets dataset. You will generate human tweets in the style of that dataset. Any events referenced in tweets, if applicable, should take place between 2015 to 2018. The tweets do not have to be factual. ",
        examples)
    print(examples)
    generated_tweets = prompt_with_few_shot_examples(
        messages=messages,
        model=DEEPSEEK_V3,
        temperature=1,
        top_p=1,
        max_completion_tokens=50,
        number_of_responses=1
    )
    print(generated_tweets)
