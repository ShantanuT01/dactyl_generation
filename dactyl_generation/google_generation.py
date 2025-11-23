"""
This module helps in generating texts using the Gemini API.
"""
from google import genai
from google.genai import types
from dactyl_generation.constants import *
from dactyl_generation.batchclient import  BatchClient
import pandas as pd
import json



class GoogleClient(BatchClient):
    def __init__(self, api_key: str):
        """
        Constructor for Google's Gemini API.

        Args:
            api_key: Gemini API key.
        """
        super().__init__()
        self.client = genai.Client(api_key=api_key)

    def upload_prompts(self, jsonl_path: str, display_name:str) -> genai.types.File:
        """
        Uploads JSONL file to Google Cloud.
        
        Args:
            jsonl_path: Local path to JSONL file containing prompts.
            display_name: Name of file on Google cloud to upload to.

        Returns:
            uploaded_file: Uploaded file object
        """
        return self.client.files.upload(file=jsonl_path, config=types.UploadFileConfig(display_name=display_name,mime_type="jsonl"))

    def create_batch_job(self, jsonl_path: str, jsonl_display_name: str, model: str, batch_display_name: str) -> dict:
        """
        Creates and starts batch job with the Gemini API. 

        Args:
            jsonl_path: Local path to JSONL file containing prompts.
            jsonl_display_name: Name of file on Google cloud to upload to.
            model: Name of LLM to use.
            batch_display_name: Batch display name to show.

        Returns:
            batch_info: Dictionary containing batch information.
        """
        uploaded_file = self.upload_prompts(jsonl_path, jsonl_display_name)
        prompts_df = pd.read_json(jsonl_path,lines=True)
        batch_job = self.client.batches.create(model=model, src=uploaded_file.name, config={DISPLAY_NAME: batch_display_name})
        batch_name = batch_job.name
        return {
            BATCH: batch_name,
            INPUT_FILE: prompts_df.to_dict(orient="records"),
            API_CALL: GEMINI
        }

    def get_batch_job_output(self, file_path: str) -> pd.DataFrame:
        """
        Fetches batch inference results and returns as pandas DataFrame.
        
        Args:
            file_path: JSON file containing object returned by `create_batch_job function`

        Returns:
            dataframe: Results containing responses and prompts.

        """
        with open(file_path, 'r') as f:
            batch_info = json.load(f)
        batch_job = self.client.batches.get(name = batch_info[BATCH])
        prompts_df = pd.DataFrame(batch_info[INPUT_FILE])
        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

            # If batch job was created with a file
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                file_content = self.client.files.download(file=result_file_name)
                lines = file_content.decode('utf-8').splitlines()
                responses = [json.loads(line.strip()) for line in lines if line.strip()]
                normalized_rows = list()
                for response in responses:
                    row = {KEY: response[KEY]}
                    obj = response[RESPONSE]
                    row[TEXT] = obj[CANDIDATES][0][CONTENT][PARTS][0][TEXT]
                    for key in obj:
                        if key == CANDIDATES:
                            continue
                        else:
                            row[key] = obj[key]
                    normalized_rows.append(row)
                results = pd.DataFrame(normalized_rows)
                results = results.merge(prompts_df, how="left",on=KEY)
                return results
        return pd.DataFrame()







