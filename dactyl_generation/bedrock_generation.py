"""
Generates texts using AWS Bedrock APIs.
!!! note
    Only supports AWS region US East 1!
"""
from litellm import completion
from typing import List
from dactyl_generation.openai_generation import format_message_with_few_shot_examples
import os
os.environ['AWS_REGION']='us-east-1'

def prompt_with_few_shot_examples(system_prompt: str, examples: List[str], model: str, temperature: float, top_p: float, max_completion_tokens: int =512) -> str:
    """
    Prompt AWS Bedrock model with few shot learning examples.

    Args:
        system_prompt: The system prompt string.
        examples: The list of examples to pass
        model: name of model
        temperature: temperature parameter
        top_p: top p parameter
        max_completion_tokens: maximum number of tokens for completion

    Returns:
        response_content: string containing message content
    """

    messages = format_message_with_few_shot_examples(system_prompt, examples)
    response = completion(model, messages, temperature=temperature, top_p=top_p,max_completion_tokens=max_completion_tokens)
    return response.choices[0].message.content


