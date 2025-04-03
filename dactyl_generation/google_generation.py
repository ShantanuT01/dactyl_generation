"""
This module helps in generating texts using the Gemini API.
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv
import typing_extensions as typing
from dactyl_generation.constants import *
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class GeneratedResponse(typing.TypedDict):
    text: str

def prompt_with_few_shot_examples(system_prompt: str, prompt: str, model_name: str, temperature: float, top_p: float, max_completion_tokens: int) -> str:
    """
    Prompt Gemini model with few shot examples. The `prompt_str` should contain the formatted examples.

    Args:
        system_prompt (str): System prompt.
        prompt (str): Prompt to pass in.
        model_name (str): Name of model.
        temperature (float): Temperature to pass.
        top_p (float): Top-p value.
        max_completion_tokens: maximum number of tokens to generate

    Returns:
        text: Generation output.
    """

    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    generation_config = genai.GenerationConfig(temperature=temperature, top_p=top_p, response_mime_type="application/json",max_output_tokens=max_completion_tokens,response_schema=GeneratedResponse)
    response = model.generate_content(prompt, generation_config=generation_config, safety_settings={
        'HATE': BLOCK_NONE,
        'HARASSMENT': BLOCK_NONE,
        'SEXUAL' : BLOCK_NONE,
        'DANGEROUS' : BLOCK_NONE
    })
    return json.loads(response.text)[TEXT]



