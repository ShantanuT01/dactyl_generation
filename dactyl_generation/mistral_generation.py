"""
Generates texts with using the Mistral Batch API.
"""

import mistralai.files
from mistralai import Mistral, File, BatchJobsOut
from io import BytesIO
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime, timezone

from dactyl_generation.batchclient import BatchClient
from dactyl_generation.constants import *


class MistralClient(BatchClient):
    def __init__(self, api_key: str):
        """
        Constructor for Mistral client.

        Args:
            api_key:
        """
        super().__init__()
        self.client = Mistral(api_key=api_key)


    def create_message_batch(self, file_name: str, prompts_df: pd.DataFrame) -> Tuple[List[dict], mistralai.models.UploadFileOut]:
        """
        Creates batch of messages to send to Mistral API.


        Args:
            file_name: Name of file in Mistral API to save as.
            prompts_df: DataFrame containing prompts and generation parameters

        Returns:
            tuple: List of requests sent, UploadFileOut object
        """

        buffer = BytesIO()
        list_of_requests = list()
        messages = prompts_df.drop(columns=[CUSTOM_ID]).to_dict(orient="records")
        for index, message_batch in enumerate(messages):
            request = {
                CUSTOM_ID: prompts_df[CUSTOM_ID].values[index],
                BODY: message_batch
            }
            list_of_requests.append(request)
            buffer.write((json.dumps(request)+"\n").encode("utf-8"))
        file = File(file_name=file_name, content=buffer.getvalue())
        return list_of_requests, self.client.files.upload(file=file, purpose=BATCH)


    def start_batch_job(self, input_file: mistralai.models.UploadFileOut, model: str) -> mistralai.models.BatchJobOut:
        """
        Start batch job from input file stored on Mistral API containing prompts.

        Args:
            input_file: input file object to create job with
            model: model name to use for generation

        Returns:
            batch_job: Batch job object
        """

        batch_job = self.client.batch.jobs.create(
            input_files=[input_file.id],
            model=model,
            endpoint="/v1/chat/completions",
            metadata={"job_type": "testing"}
        )
        return batch_job

    def create_batch_job(self, file_name: str, prompts_df: pd.DataFrame) -> dict:
        """
        Creates batch job for set of prompts given file name to save Mistral prompts to.
        
        Args:
            file_name: name of file to upload to Mistral API.
            prompts_df: DataFrame containing generation prompts and parameters.

        Returns:
            info: dictionary containing batch job info
        """
        assert(len(prompts_df[MODEL].unique()) == 1)
        model = prompts_df[MODEL].unique()[0]
        prompts, input_file = self.create_message_batch(file_name, prompts_df)
        batch_job = self.start_batch_job(input_file, model)
        input_file = input_file.model_dump(mode="json")
        batch_job = batch_job.model_dump(mode="json")
        return {"batch_job": batch_job, INPUT_FILE: input_file, PROMPTS: prompts, API_CALL: MISTRAL}



    def get_batch_jobs(self) -> BatchJobsOut:
        """
        Helper method to get status of all batch jobs.

        Returns:
            batch_jobs_list: list of all batch jobs
        """
        return self.client.batch.jobs.list(
            metadata={"job_type": "testing"}
        )


    def get_batch_job_output(self, file_path: str) -> pd.DataFrame:
        """
        Gets batch job results using saved metadata from a local JSON file.
        
        Args:
            file_path: local JSON file containing output of the `create_batch_job` function

        Returns:
            df: pandas DataFrame of generations.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        job_id = data["batch_job"]["id"]
        output_file = self.client.batch.jobs.get(job_id=job_id).output_file
        content = self.client.files.download(file_id=output_file).read().decode("utf-8")
        json_obj = "[" + ", ".join(content.splitlines()) + "]"
        responses = json.loads(json_obj)
        rows = list()
        for response in responses:
            row = dict()
            row[CUSTOM_ID] = response[CUSTOM_ID]
            row[TEXT] = response[RESPONSE][BODY][CHOICES][0][MESSAGE][CONTENT]
            row[TIMESTAMP] = str(datetime.fromtimestamp(response[RESPONSE][BODY][CREATED], tz=timezone.utc))
            rows.append(row)
        raw_prompts = pd.DataFrame([{**prompt[BODY], **{CUSTOM_ID: prompt[CUSTOM_ID]}} for prompt in data[PROMPTS]])
        generations = pd.DataFrame(rows)
        return generations.merge(raw_prompts, on=CUSTOM_ID,how="left")



