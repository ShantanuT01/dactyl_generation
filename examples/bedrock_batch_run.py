import json
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
from dactyl_generation.bedrock_generation import create_batch_job, get_batch_job_output

prompts_df = pd.read_json("local-tiny-test.json")
# add additional model info and rename prompt column
prompts_df["max_gen_len"] = 100
# AWS Bedrock needs at least 100 calls
prompts_df = pd.concat([prompts_df] * 5)

# set up file paths
input_file_path = "llama-3-3-70b-inputs.json"
output_file_path = "llama-3-3-70b-outputs.json"

if not os.path.exists(input_file_path):
    model_name = "us.meta.llama3-3-70b-instruct-v1:0"
    role_arn = os.environ["ROLE_ARN"]
    job_name = os.environ["JOB_NAME"]
    s3_input_path = os.environ["S3_INPUT_PATH"]
    s3_output_path = os.environ["S3_OUTPUT_PATH"]
    results = create_batch_job(prompts_df, s3_input_path, s3_output_path, model=model_name, role_arn=role_arn, job_name=job_name)
    with open(input_file_path, 'w+') as f:
        json.dump(results, f, indent=4)
else:
    results = get_batch_job_output(input_file_path)
    results.to_json(output_file_path,index=False,orient="records",indent=4)