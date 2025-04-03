"""
Generates texts using AWS Bedrock APIs.
!!! note
    Only supports AWS region US East 1!
"""
from litellm import completion
from typing import List

import os
os.environ['AWS_REGION']='us-east-1'

def prompt(messages:List[dict],  model: str, temperature: float, top_p: float, max_completion_tokens: int =512) -> str:
    """
    Prompt AWS Bedrock model with few shot learning examples.

    Args:
        messages: List of OpenAI messages
        model: name of model
        temperature: temperature parameter
        top_p: top p parameter
        max_completion_tokens: maximum number of tokens for completion

    Returns:
        response_content: string containing message content
    """

    response = completion(model, messages, temperature=temperature, top_p=top_p,max_completion_tokens=max_completion_tokens)
    return response.choices[0].message.content


