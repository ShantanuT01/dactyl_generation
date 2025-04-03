"""
This module helps in generating texts using the Gemini API.
"""
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import List
import typing_extensions as typing
from dactyl_generation.constants import *
import json


load_dotenv()
GOOGLE_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_SAFETY_SETTINGS = {
        'HATE': BLOCK_NONE,
        'HARASSMENT': BLOCK_NONE,
        'SEXUAL' : BLOCK_NONE,
        'DANGEROUS' : BLOCK_NONE
    }
UPDATED_SAFETY_SETTINGS = list()
for category in GEMINI_SAFETY_SETTINGS:
    UPDATED_SAFETY_SETTINGS.append(
        types.SafetySetting(
            category=category,
            threshold=BLOCK_NONE
        )
    )
class GeneratedResponse(typing.TypedDict):
    text: str

def prompt(messages: List[dict], model_name: str, temperature: float, top_p: float, max_completion_tokens: int) -> str:
    """
    Prompt Gemini model with few shot examples. The `prompt_str` should contain the formatted examples.

    Args:
        messages: List of OpenAI messages
        model_name (str): Name of model.
        temperature (float): Temperature to pass.
        top_p (float): Top-p value.
        max_completion_tokens: maximum number of tokens to generate

    Returns:
        text: Generation output.
    """

    system_instructions = list()
    user_instructions = list()
    for message in messages:
        if message[ROLE] == SYSTEM:
            system_instructions.append(message[CONTENT])
        else:
            user_instructions.append(message[CONTENT])
    prompt_config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        max_output_tokens=max_completion_tokens,
        top_p=top_p,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=GeneratedResponse,
        safety_settings=UPDATED_SAFETY_SETTINGS
    )

    response = GOOGLE_CLIENT.models.generate_content(model=model_name, contents=user_instructions, config=prompt_config)
    return response.parsed.text



