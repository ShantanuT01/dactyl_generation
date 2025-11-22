import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.google_generation import GoogleClient

client = GoogleClient(os.environ["GEMINI_API_KEY"])

# set up file paths
input_file_path = "gemini-2.5-flash-inputs.json"
output_file_path = "gemini-2.5-flash-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job("gemini-sample-prompts.jsonl","sample-batch","gemini-2.5-flash","test-batch")
    with open(input_file_path,'w+') as f:
        json.dump(results, f, indent=4)
else:
    results = client.get_batch_job_output(input_file_path)
    results.to_json(output_file_path, orient="records",indent=4)
