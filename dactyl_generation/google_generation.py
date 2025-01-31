import google.generativeai as genai
import os
from dotenv import load_dotenv
import typing_extensions as typing
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class TwitterResponse(typing.TypedDict):
    tweet: str

def prompt_with_few_shot_examples(system_prompt, prompt, model_name, temperature, top_p):
    """
    Prompt Gemini model with few shot examples.

    :param system_prompt: The system prompt to be sent
    :param prompt: Prompt request
    :param model_name: name of model
    :param temperature: temperature parameter
    :param top_p: top p parameter
    :return: text of generated tweet
    """
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    generation_config = genai.GenerationConfig(temperature=temperature, top_p=top_p, response_mime_type="application/json",response_schema=TwitterResponse)
    response = model.generate_content(prompt, generation_config=generation_config, safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL' : 'BLOCK_NONE',
        'DANGEROUS' : 'BLOCK_NONE'
    })
    return json.loads(response.text)["tweet"]



