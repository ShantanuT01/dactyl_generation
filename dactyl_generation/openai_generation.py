"""
Generates texts with using the OpenAI Batch API.
"""

from openai import OpenAI
from dactyl_generation.constants import *
from dactyl_generation.batchclient import BatchClient
import pandas as pd
import json
from io import BytesIO
from datetime import datetime, timezone



class OpenAIClient(BatchClient):
    def __init__(self, api_key: str) -> None:
        """
        Constructor for OpenAI Client key.
        Args:
            api_key: OpenAI API key.
        """
        super().__init__()
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def create_individual_request(custom_id: str, message_body: dict) -> dict:
        """
        Creates OpenAI REST API request for a single request.
        Args:
            custom_id: Custom ID of request
            message_body: dictionary of a single message. This includes the messages, max_completion_token parameters etc.

        Returns:
            request: individual request formatted for OpenAI REST API.
        """
        request = {CUSTOM_ID: str(custom_id), "method": "POST", "url": "/v1/chat/completions", BODY: message_body}
        return request


    def create_batch_job(self, prompts_df: pd.DataFrame) -> dict:
        """
           Creates batch job of prompts given messages and temperatures.

           Args:
               prompts_df: DataFrame where each row corresponds to an OpenAI API call.

           Returns:
               results: dictionary containing request information
           """
        json_strs = list()
        requests = list()
        records = prompts_df.drop(columns=[CUSTOM_ID]).to_dict("records")
        for i, record in enumerate(records):
            request = OpenAIClient.create_individual_request(prompts_df[CUSTOM_ID].values[i], record)
            requests.append(request)
            json_strs.append(json.dumps(request))
        buffer = BytesIO(("\n".join(json_strs)).encode("utf-8"))
        # with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as fp:
        #    fp.write("\n".join(json_strs))
        #    temp_filename = fp.name

        batch_file = self.client.files.create(
            file=buffer,
            purpose="batch"
        )
        #  os.remove(temp_filename)

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        result_file_id = batch_job.id

        return {
            RESULT_FILE_ID: result_file_id,
            INPUT_FILE: requests,
            API_CALL: OPENAI
        }

    def get_batch_job_output(self, file_path: str) -> pd.DataFrame:
        """
        Gets batch job results using saved metadata from a local JSON file.
        Args:
            file_path: local JSON file containing output of the `create_batch_job` function

        Returns:
            df: pandas DataFrame of generations.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        batch_job = self.client.batches.retrieve(data[RESULT_FILE_ID])
        result = self.client.files.content(batch_job.output_file_id).content
        df = pd.read_json(BytesIO(result), lines=True)
        responses = df[RESPONSE]
        custom_ids = df[CUSTOM_ID]
        generations = list()
        for response, custom_id in zip(responses, custom_ids):
            generation = dict()
            generation[TEXT] = response[BODY][CHOICES][0][MESSAGE][CONTENT]
            generation[CUSTOM_ID] = custom_id
            generation[TIMESTAMP] = str(datetime.fromtimestamp(response[BODY][CREATED], tz=timezone.utc))
            generations.append(generation)
        generations = pd.DataFrame(generations)
        requests = pd.DataFrame(data[INPUT_FILE])

        generations = generations.merge(requests, on=CUSTOM_ID, how='left')
        return generations














