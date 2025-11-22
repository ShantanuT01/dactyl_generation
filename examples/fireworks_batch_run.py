import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.fireworks_generation import FireworksAIClient


client = FireworksAIClient(os.environ["FIREWORKS_API_KEY"], os.environ["FIREWORKS_ACCOUNT_ID"])

# set up file paths
input_file_path = "qwen3-235b-inputs.json"
output_file_path = "qwen3-235b-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job("sample-batch","accounts/fireworks/models/qwen3-235b-a22b-instruct-2507","fireworks_prompts.jsonl","sample-input-dataset","sample-output-dataset",{})
    with open(input_file_path,'w+') as f:
        json.dump(results, f, indent=4)
else:
    client.get_batch_job_output(input_file_path).to_json(output_file_path, orient="records",indent=4)
    #results.to_json(output_file_path,index=False,orient="records",indent=4)