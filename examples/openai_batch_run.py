import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.openai_generation import OpenAIClient

prompts_df = pd.read_json("local-tiny-test.json")
# add additional model info and rename prompt column
prompts_df["model"] = "gpt-5-mini-2025-08-07"
#prompts_df["frequency_penalty"] = 1.1
prompts_df = prompts_df.drop(columns=["temperature","top_p"])
prompts_df = prompts_df.rename(columns={"prompt":"messages"})
client = OpenAIClient(os.environ["OPENAI_API_KEY"])
# set up file paths
input_file_path = "gpt-5-mini-inputs.json"
output_file_path = "gpt-5-mini-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job(prompts_df)
    with open(input_file_path,'w+') as f:
        json.dump(results, f, indent=4)
else:
    results = client.get_batch_job_output(input_file_path)
    results.to_json(output_file_path,index=False,orient="records",indent=4)