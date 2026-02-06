# Batch Inference with Together AI

This short tutorial explains how to use `dactyl_generation.together_generation` to run a batch job with Together AI models and save the results locally.

---

## Prerequisites

```bash
pip install dactyl-generation
```

Create a `.env` file with your Together AI credentials:

```bash
TOGETHER_API_KEY=fw-xxxxxxxxxxxxxxxxxxxx
```

Your input file must be a JSONL file compatible with Fireworks batch prompts, for example (taken from the Fireworks AI documentation):

```json title="together_prompts.jsonl"
{"custom_id": "request-1", "body": {"model": "deepseek-ai/DeepSeek-V3", "messages": [{"role": "user", "content": "Hello, world!"}], "max_tokens": 200}}
{"custom_id": "request-2", "body": {"model": "deepseek-ai/DeepSeek-V3", "messages": [{"role": "user", "content": "Explain quantum computing"}], "max_tokens": 200}}
```

---

## Creating and Fetching Batch Job Results

Instantiate the client.

```python
import dotenv
dotenv.load_dotenv()

from dactyl_generation.together_generation import TogetherClient

client = TogetherClient(
    os.environ["TOGETHER_API_KEY"]
)
```

Define local cache files.

```python
input_file_path = "deepseek-inputs.json"
output_file_path = "deepseek-outputs.json"
```

### Create batch job (first run)

We leave the `inference_parameters` as empty, but feel free to use them. 
```python
results = client.create_batch_job("together_prompts.jsonl")

with open(input_file_path, "w+") as f:
    json.dump(results, f, indent=4)
```

### Fetch results (subsequent runs)

```python
client.get_batch_job_output(input_file_path).to_json(
    output_file_path,
    orient="records",
    indent=4
)
```

---

## Full Script

```python
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

```


