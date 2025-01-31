from litellm import completion
from dactyl_generation.openai_generation import format_message_with_few_shot_examples
import os
os.environ['AWS_REGION']='us-east-1'

def prompt_with_few_shot_examples(system_prompt, examples, model, temperature, top_p, max_completion_tokens=512):
    """
    Prompt AWS Bedrock model with few shot learning examples.

    :param system_prompt: The system prompt string.
    :param examples: The list of examples to pass
    :param model: name of model
    :param temperature: temperature parameter
    :param top_p: top p parameter
    :param max_completion_tokens: maximum number of tokens for completion
    :return: generated text in a list
    """
    messages = format_message_with_few_shot_examples(system_prompt, examples)
    response = completion(model, messages, temperature=temperature, top_p=top_p,max_completion_tokens=max_completion_tokens)
    return response.choices[0].message.content


