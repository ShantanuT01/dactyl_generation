from dactyl_generation.constants import *
from dactyl_generation.batchclient import BatchClient
from typing import Dict
import json
from together import Together
import pandas as pd


class TogetherClient(BatchClient):
    def __init__(self, api_key: str) -> None:
        """
        Constructor for Together AI client. 

        Args:
            api_key (str): API key
        """
        super().__init__()
        self.client = Together(api_key=api_key)
    
    def create_batch_job(self, jsonl_path: str) -> Dict[str, str]:
        """
        Creates batch job from JSONL file.

        Args:
            jsonl_path: Path to JSONL file.

        Returns:
            batch_job_input: Dictionary of batch job information.
        """
        file_resp = self.client.files.upload(
            file=jsonl_path, purpose="batch-api", check=False
        )
        prompts_df = pd.read_json(jsonl_path, lines=True)
        file_id = file_resp.id

        batch = self.client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions").job


        return {
            BATCH_ID: batch.id,
            INPUT_FILE: batch.input_file_id,
            CREATED: str(batch.created_at),
            MODEL: batch.x_model_id,
            PROMPTS: prompts_df.to_dict(orient="records")
        }

    def get_batch_job_output(self, batch_job_file: str) -> pd.DataFrame:
        """
        Retrieves batch job output.


        Args:
            batch_job_file: Path to batch job input, generated from the function `create_batch_job`.
        Returns:
            result_df: Pandas DataFrame of batch job output.
        """
        with open(batch_job_file, 'r') as f:
            data = json.load(f)

        batch_id = data[BATCH_ID]
        batch = self.client.batches.retrieve(batch_id)
        prompts_df = pd.DataFrame(data[PROMPTS])


        with self.client.files.with_streaming_response.content(id=batch.output_file_id) as response:
            results_df = pd.DataFrame([json.loads(line) for line in response.iter_lines()])

        prompts_df = prompts_df.merge(results_df, how="left", on=CUSTOM_ID)
        return prompts_df
