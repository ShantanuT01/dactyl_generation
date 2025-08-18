"""
Generates texts using AWS Bedrock APIs.
!!! note
    Only supports AWS region US East 1!
"""
from litellm import completion
from typing import List
import os
import pandas as pd

from dactyl_generation.constants import *
os.environ['AWS_REGION']='us-east-1'
import boto3
import json
import datetime


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


def format_llama_prompt(messages: List[dict]) -> str:
    """
    Formats OpenAI style message to Llama 3.2 style.
    Args:
        messages: list of dictionaries containing OpenAI style messages

    Returns:
        llama_prompt: formatted llama prompt
    """
    formatted_prompt = "<|begin_of_text|>"
    for message in messages:
        role =  message[ROLE]
        formatted_prompt += LLAMA_START_HEADER + role + LLAMA_END_HEADER + message[CONTENT] + "<|eot_id|>"
    formatted_prompt += f"{LLAMA_START_HEADER}assistant{LLAMA_END_HEADER}"
    return formatted_prompt


def create_jsonl_input_for_llama(prompts_df: pd.DataFrame, s3_path: str, max_gen_len:int = 512) -> pd.DataFrame:
    """
    Creates a JSONL file to upload to S3.
    Args:
        prompts_df: prompt dataframe containing OpenAI style messages
        s3_path: Path to S3 bucket to save file
        max_gen_len: maximum generation token count per request

    Returns:
        None
    """
    messages = prompts_df[PROMPT].to_list()
    temperatures = prompts_df[TEMPERATURE].to_list()
    top_ps = prompts_df[TOP_P].to_list()
    rows = list()
    for i in range(len(messages)):
        rows.append({
            RECORDID: f"CALL{str(i).zfill(7)}",
            MODELINPUT:{
                PROMPT: format_llama_prompt(messages[i]),
                TEMPERATURE: temperatures[i],
                TOP_P: top_ps[i],
                "max_gen_len": max_gen_len
            }
        }
        )
    input_frame = pd.DataFrame(rows)
    input_frame.to_json(s3_path, orient="records",index=False, lines=True)
    prompts_df_ret = pd.DataFrame(prompts_df)
    prompts_df_ret[RECORDID] = input_frame[RECORDID].to_list()
    return prompts_df_ret


def create_batch_job(prompts_df: pd.DataFrame, s3_input_path: str, s3_output_path: str, model: str, role_arn: str, job_name: str, max_tokens: int = 512) -> dict:
    """
    Creates batch job for Bedrock models.

    !!! warning
        This function has not been tested yet!

    Args:
        prompts_df: Dataframe of OpenAI-style prompts.
        s3_input_path: Input data path.
        s3_output_path: Output data path.
        model: Bedrock model ID.
        role_arn: Role to run batch job.
        job_name: Name of job

    Returns:
        jobArn: dictionary containing single string
    """
    inputted_frame = create_jsonl_input_for_llama(prompts_df, s3_input_path, max_gen_len=max_tokens)
    bedrock = boto3.client(service_name="bedrock",region_name="us-east-1")
    input_data_config = (
        {
            S3_INPUT_DATA_CONFIG: {
                S3URI: s3_input_path
            }
        }
    )
    output_data_config = (
        {
            S3_OUTPUT_DATA_CONFIG:{
                S3URI: s3_output_path
            }
        }
    )

    response = bedrock.create_model_invocation_job(
        roleArn=role_arn,
        modelId=model,
        jobName=job_name,
        inputDataConfig=input_data_config,
        outputDataConfig=output_data_config
    )
    inputted_frame[MODEL] = model
    return {
        JOB_ARN: response.get(JOB_ARN),
        S3_OUTPUT_DATA_CONFIG: s3_output_path,
        API_CALL: BEDROCK,
        JOB_NAME: job_name,
        INPUT_FILE: json.loads(inputted_frame.to_json(orient='records'))

    }


def get_batch_job_output(file_path: str) -> pd.DataFrame:
    """
    Fetches batch job results given JSON file.
    Args:
        file_path: JSON file containing jobArn.

    Returns:

    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    job_arn = data[JOB_ARN].split("/")[-1]
    s3_output_path = os.path.join(data[S3_OUTPUT_DATA_CONFIG],job_arn, data[JOB_NAME]+".jsonl.out").replace("\\","/")
    s3_client = boto3.resource('s3')
    # ignore s3://
    bucket_name = data[S3_OUTPUT_DATA_CONFIG].split("/")[2]
    bucket = s3_client.Bucket(bucket_name)
    folder_path = "/".join(data[S3_OUTPUT_DATA_CONFIG].split("/")[3:]) + job_arn + "/"
    target_file = None
    for object_summary in bucket.objects.filter(Prefix=folder_path):
        if object_summary.key.endswith(".jsonl.out"):
            target_file = object_summary.key
            break


    if target_file:
        output_df = pd.read_json(f"s3://{bucket_name}/"+target_file, lines=True)
        rows = list()
        for _, row in output_df.iterrows():
            entry = dict()
            entry[TEXT] = row[MODEL_OUTPUT][GENERATION].strip()
            entry[RECORDID] = row[RECORDID]
            rows.append(entry)

        outputs = pd.DataFrame(rows)
        inputs = pd.DataFrame(data[INPUT_FILE])
        outputs = outputs.merge(inputs, how='left', on=RECORDID)
        outputs = outputs.drop(columns=RECORDID)
        return outputs
    else:
        raise Exception(f"{bucket_name} does not contain .jsonl.out file! Please check if job has completed.")





