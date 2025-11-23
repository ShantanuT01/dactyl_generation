import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.mistral_generation import MistralClient

prompts_df = pd.read_json("local-tiny-test.json")
client = MistralClient(os.environ["MISTRAL_API_KEY"])
# add additional model info and rename prompt column
prompts_df["model"] = "mistral-small-latest"
prompts_df["frequency_penalty"] = 1.1
prompts_df = prompts_df.rename(columns={"prompt":"messages"})

# set up file paths
input_file_path = "mistral-small-latest-inputs.json"
output_file_path = "mistral-small-latest-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job("sample",prompts_df)
    with open(input_file_path,'w+') as f:
        json.dump(results, f, indent=4)
else:
    results = client.get_batch_job_output(input_file_path)
    results.to_json(output_file_path,index=False,orient="records",indent=4)