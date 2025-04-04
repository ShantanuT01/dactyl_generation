"""
This module helps in generating texts using the Gemini API.

!!! danger
    For Gemini, all safety filters have been turned off!
"""
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import List
import typing_extensions as typing
from dactyl_generation.constants import *
from pydantic import BaseModel


load_dotenv()
GOOGLE_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold=BLOCK_NONE
    )
]

class GeneratedResponse(BaseModel):
    text: str

def prompt(messages: List[dict], model_name: str, temperature: float, top_p: float, max_completion_tokens: int) -> str:
    """
    Prompt Gemini model with an individual request.

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
        safety_settings=GEMINI_SAFETY_SETTINGS
    )

    response = GOOGLE_CLIENT.models.generate_content(model=model_name, contents=user_instructions, config=prompt_config)
    return response.parsed[TEXT]



