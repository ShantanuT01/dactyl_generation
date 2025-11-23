"""
Generates texts using Fireworks AI API.
"""
from dactyl_generation.constants import *
import pandas as pd
import requests
import json
from dactyl_generation.batchclient import BatchClient
from datetime import datetime, timezone

class FireworksAIClient(BatchClient):
    def __init__(self, fireworks_api_key: str, account_id: str):
        """
        Constructor for Fireworks AI Client.

        Args:
            fireworks_api_key: Fireworks API key.
            account_id: Account to use.
        """
        super().__init__()
        self.fireworks_api_key = fireworks_api_key
        self.authorization_headers = {
            "Authorization": f"Bearer {self.fireworks_api_key}",
        }
        self.account_id = account_id

    def create_dataset(self, jsonl_path: str, dataset_id: str) -> None:
        """
        Creates and uploads a dataset to the Fireworks AI datasets endpoints.

        Args:
            jsonl_path: Path to local dataset of prompts.
            dataset_id: Name of dataset to upload as

        Returns:

        """
        data_args = {"datasetId": dataset_id, "dataset": {"userUploaded":{}}}
        response_for_dataset_initialization = requests.post(f"https://api.fireworks.ai/v1/accounts/{self.account_id}/datasets",headers=self.authorization_headers, json=data_args)
        response_for_dataset_initialization.raise_for_status()

        with open(jsonl_path, 'rb') as f:
            files = {"file": f}
            response_for_dataset_upload = requests.post(f"https://api.fireworks.ai/v1/accounts/{self.account_id}/datasets/{dataset_id}:upload", headers=self.authorization_headers, files=files)
            response_for_dataset_upload.raise_for_status()




    def create_batch_job(self, batch_id: str, model: str, prompts_path: str, input_dataset_id: str, output_dataset_id: str, inference_parameters: dict) -> dict:
        """
        Creates a batch job for Fireworks AI Client.

        Args:
            batch_id: Name of batch job.
            model: Fireworks model.
            prompts_path: JSONL path to prompts to upload.
            input_dataset_id: Input dataset name for prompts.
            output_dataset_id: Output dataset name for generations.
            inference_parameters: Additional inference parameters.

        Returns:
            batch_dict: Dictionary containing batch information if successful.

        """
        prompts_df = pd.read_json(prompts_path, lines=True)
        rows = list()
        for _, row in prompts_df.iterrows():
            new_row = {CUSTOM_ID: row[CUSTOM_ID]}
            new_row.update(row[BODY])
            rows.append(new_row)
        to_save_prompts = pd.DataFrame(rows)
        self.create_dataset(prompts_path, input_dataset_id)

        data = {
            MODEL: model,
            INPUT_DATASET_ID:f"accounts/{self.account_id}/datasets/{input_dataset_id}",
            OUTPUT_DATASET_ID: f"accounts/{self.account_id}/datasets/{output_dataset_id}",
            "inferenceParameters": inference_parameters
        }

        response = requests.post(f"https://api.fireworks.ai/v1/accounts/{self.account_id}/batchInferenceJobs?batchInferenceJobId={batch_id}",headers=self.authorization_headers, json=data)
        response.raise_for_status()
        return {"batch_job": batch_id, INPUT_DATASET_ID: input_dataset_id, OUTPUT_DATASET_ID: output_dataset_id, PROMPTS: to_save_prompts.to_dict(orient="records"), API_CALL: FIREWORKS_AI}



    def get_batch_job_output(self, file_path: str) -> pd.DataFrame:
        """
        Fetches batch job results given JSON path to object generated from `create_batch_job` function.
        
        Args:
            file_path: File path to JSON object from `create_batch_job` function.

        Returns:
            dataframe: Pandas dataframe containing generations.
        """
        with open(file_path,'r') as f:
            batch_request = json.load(f)

        prompts_df = pd.DataFrame(batch_request[PROMPTS])
        output_dataset = batch_request[OUTPUT_DATASET_ID]
        response = requests.get(f"https://api.fireworks.ai/v1/accounts/{self.account_id}/datasets/{output_dataset}:getDownloadEndpoint", headers=self.authorization_headers)
        response = response.json()
        files = response["filenameToSignedUrls"]
        outputs = pd.read_json(files[f"dataset/{batch_request[OUTPUT_DATASET_ID]}/BIJOutputSet.jsonl"], lines=True)
        normalized_rows = list()
        for _, row in outputs.iterrows():
            new_row = {CUSTOM_ID: row[CUSTOM_ID],
                       TIMESTAMP: str(datetime.fromtimestamp(row[RESPONSE][CREATED], tz=timezone.utc)),
                       TEXT: row[RESPONSE][CHOICES][0][MESSAGE][CONTENT],
                       USAGE: row[RESPONSE][USAGE],
                       MODEL: row[RESPONSE][MODEL],
                       ID: row[RESPONSE][ID]
            }
            normalized_rows.append(new_row)
        normalized_df = pd.DataFrame(normalized_rows)
        normalized_df = normalized_df.merge(prompts_df,how="left",on=CUSTOM_ID)
        return normalized_df

