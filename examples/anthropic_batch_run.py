import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.anthropic_generation import create_batch_job, get_batch_job_output

prompts_df = pd.read_json("local-tiny-test.json")
# add additional model info
prompts_df["model"] = "claude-3-5-haiku-20241022"
prompts_df["max_tokens"] = 100
prompts_df["top_k"] = 300


# set up file paths
input_file_path = "claude-3-5-haiku-inputs.json"
output_file_path = "claude-3-5-haiku-outputs.json"

if not os.path.exists(input_file_path):
    results = create_batch_job(prompts_df)
    with open(input_file_path,'w+') as f:
        json.dump(results, f, indent=4)
else:
    results = get_batch_job_output(input_file_path)
    results.to_json(output_file_path,index=False,orient="records",indent=4)