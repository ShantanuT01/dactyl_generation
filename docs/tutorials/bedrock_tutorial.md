# Batch Inference with AWS Bedrock

## Prerequisites

Before running the script, make sure you have:

1. **AWS credentials** configured that allow you to assume a Bedrock role and access the relevant S3 buckets.
2. An IAM role ARN that can be used for Bedrock:
   - This will be referenced as `ROLE_ARN` in your environment variables.
3. An S3 bucket/folder where:
   - Input payloads for the batch job will be stored (`S3_INPUT_PATH`)
   - Output results from the batch job will be written (`S3_OUTPUT_PATH`)
4. A small local JSON file (e.g. `local-tiny-test.json`) containing your prompts.

Example `local-tiny-test.json` (very minimal):

```json
 [
  {
    "recordId":"custom_id_0",
    "prompt":[
            {
                "role":"system",
                "content":"You will write a human-like abstract for the paper with the following title. The user will give you an example abstract to mimic its style. Output only the new abstract, nothing else.\/nPaper Title: Shearing Mechanisms of Co-Precipitates in IN718"
            },
            {
                "role":"user",
                "content":"The electric, magnetic, and thermal properties of three perovskite cobaltites\nwith the same 30% hole doping and ferromagnetic ground state were investigated\ndown to very low temperatures. With decreasing size of large cations, the\nferromagnetic Curie temperature and spontaneous moments of cobalt are gradually\nsuppressed - $T_C=130$ K, 55 K and 25 K and $m = 0.68 \\mu_B$, 0.34 $\\mu_B$ and\n0.23 $\\mu_B$ for Nd$_{0.7}$Sr$_{0.3}$CoO$_3$, Pr$_{0.7}$Ca$_{0.3}$CoO$_3$ and\nNd$_{0.7}$Ca$_{0.3}$CoO$_3$, respectively. The moment reduction with respect to\nmoment of the conventional ferromagnet La$_{0.7}$Sr$_{0.3}$CoO$_3$ ($T_C=230$\nK, $m = 1.71 \\mu_B$) in so-called IS\/LS state for Co$^{3+}$\/Co$^{4+}$, was\noriginally interpreted using phase-separation scenario. Based on the present\nresults, mainly the analysis of Schottky peak originating in Zeeman splitting\nof the ground state Kramers doublet of Nd$^{3+}$, we find, however, that\nferromagnetic phase in Nd$_{0.7}$Ca$_{0.3}$CoO$_3$ and likely also\nPr$_{0.7}$Ca$_{0.3}$CoO$_3$ is uniformly distributed over all sample volume,\ndespite the severe drop of moments. The ground state of these compounds is\nidentified with the LS\/LS-related phase derived theoretically by Sboychakov\n\\textit{et al.} [Phys. Rev. B \\textbf{80}, 024423 (2009)]. The ground state of\nNd$_{0.7}$Sr$_{0.3}$CoO$_3$ with an intermediate cobalt moment is inhomogeneous\ndue to competing of LS\/LS and IS\/LS phases. In the theoretical part of the\nstudy, the crystal field split levels for $4f^3$ (Nd$^{3+}$), $4f^2$\n(Pr$^{3+}$) and $4f^1$ (Ce$^{3+}$ or Pr$^{4+}$) are calculated and their\nmagnetic characteristics are presented."
            }
        ],
        "temperature":0.3221183249,
        "top_p":0.8501584248
    }
]
````

Install the `dactyl-generation` library. 

```bash
pip install dactyl-generation
```


Make a `.env` file containing the following variables with your values.

```bash
ROLE_ARN=arn:aws:iam::<your-account-id>:role/<your-bedrock-role>
JOB_NAME=llama3-3-70b-batch-test
S3_INPUT_PATH=s3://your-bucket/path/to/inputs/
S3_OUTPUT_PATH=s3://your-bucket/path/to/outputs/
```

**Notes:**

* `ROLE_ARN`: IAM role to assume for Bedrock.
* `JOB_NAME`: A unique identifier for this particular Bedrock batch job.
* `S3_INPUT_PATH`: S3 location where the input payloads for the batch job will be stored.
* `S3_OUTPUT_PATH`: S3 location where Bedrock will write generation outputs.

## Creating and Fetching Batch Job Results

```python
import dotenv
dotenv.load_dotenv()

import json
import os
import dotenv
import pandas as pd



# Load .env file (ROLE_ARN, JOB_NAME, S3 paths, etc.)
dotenv.load_dotenv()

# Load prompts from a local JSON file
prompts_df = pd.read_json("local-tiny-test.json")

# Add additional model configuration
prompts_df["max_gen_len"] = 100

# AWS Bedrock needs at least 100 calls, so for our test, we 
# just duplicate the first hundred
prompts_df = pd.concat([prompts_df] * 100, ignore_index=True)

# Assign a unique recordId for each row
prompts_df["recordId"] = [str(x) for x in range(len(prompts_df))]
```

We create a `BedrockClient` using a role ARN stored in the environment:

```python
from dactyl_generation.bedrock_generation import BedrockClient
client = BedrockClient(role_arn=os.environ["ROLE_ARN"])
```

We’ll store the job metadata and final results locally in JSON files:

```python
# Local file paths
input_file_path = "llama-3-3-70b-inputs.json"
output_file_path = "llama-3-3-70b-outputs.json"
```

These act as caching/checkpoint files:

* `llama-3-3-70b-inputs.json`: Information returned by `create_batch_job` (e.g. job ID, S3 paths).
* `llama-3-3-70b-outputs.json`: The final parsed outputs of the batch job.


We first create the batch job on AWS Bedrock and save the job metadata locally. 
```python

model_name = "us.meta.llama3-3-70b-instruct-v1:0"

job_name = os.environ["JOB_NAME"]
s3_input_path = os.environ["S3_INPUT_PATH"]
s3_output_path = os.environ["S3_OUTPUT_PATH"]

# Submit the batch job to Bedrock
results = client.create_batch_job(
    prompts_df,
    s3_input_path,
    s3_output_path,
    model=model_name,
    job_name=job_name,
)

# Save job metadata so we can reuse it later
with open(input_file_path, "w+") as f:
    json.dump(results, f, indent=4)
```

At this stage, Bedrock is processing your batch. Depending on the size, it can take some time to complete.



```python
# Read outputs for an existing batch job and convert to DataFrame
results = client.get_batch_job_output(input_file_path)

# Save outputs as JSON (an array of records)
results.to_json(
    output_file_path,
    index=False,
    orient="records",
    indent=4,
)
```


## Full Script

For convenience, here is the full script as a single block:

```python
import json
import os
import dotenv
import pandas as pd

from dactyl_generation.bedrock_generation import BedrockClient

# Load environment variables from .env
dotenv.load_dotenv()

# Load prompts from local JSON file
prompts_df = pd.read_json("local-tiny-test.json")

# Add generation configuration
prompts_df["max_gen_len"] = 100

# Ensure we have at least 100 calls for AWS Bedrock
prompts_df = pd.concat([prompts_df] * 100, ignore_index=True)

# Assign unique record IDs
prompts_df["recordId"] = [str(x) for x in range(len(prompts_df))]

# Initialize the Bedrock client
client = BedrockClient(role_arn=os.environ["ROLE_ARN"])

# Local cache/checkpoint file paths
input_file_path = "llama-3-3-70b-inputs.json"
output_file_path = "llama-3-3-70b-outputs.json"

if not os.path.exists(input_file_path):
    # First run: submit a new batch job
    model_name = "us.meta.llama3-3-70b-instruct-v1:0"

    job_name = os.environ["JOB_NAME"]
    s3_input_path = os.environ["S3_INPUT_PATH"]
    s3_output_path = os.environ["S3_OUTPUT_PATH"]

    # Create batch job on Bedrock
    results = client.create_batch_job(
        prompts_df,
        s3_input_path,
        s3_output_path,
        model=model_name,
        job_name=job_name,
    )

    # Save job metadata to disk
    with open(input_file_path, "w+") as f:
        json.dump(results, f, indent=4)

else:
    # Subsequent run: fetch completed outputs
    results = client.get_batch_job_output(input_file_path)

    # Save outputs to a JSON file
    results.to_json(
        output_file_path,
        index=False,
        orient="records",
        indent=4,
    )
```




