from dactyl_generation.constants import *
from dactyl_generation.batchclient import BatchClient

from together import Together

class TogetherClient(BatchClient):
    def __init__(self, api_key: str) -> None:
        """
        Constructor for Together AI client. 

        Args:
            api_key (str): API key
        """
        self.client = Together(api_key=api_key)
    
    def create_batch_job(self, jsonl_path: str):
        pass

