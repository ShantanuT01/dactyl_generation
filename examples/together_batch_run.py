

import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.together_generation import TogetherClient


client = TogetherClient(os.environ["TOGETHER_API_KEY"])

# set up file paths
input_file_path = "deepseek-inputs.json"
output_file_path = "deepseek-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job("together_prompts.jsonl")
    with open(input_file_path,'w+') as f:
        json.dump(results, f, indent=4)
else:
    client.get_batch_job_output(input_file_path).to_json(output_file_path, orient="records",indent=4)
