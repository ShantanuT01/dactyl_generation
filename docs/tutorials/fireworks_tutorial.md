# Batch Inference with Fireworks AI

This short tutorial explains how to use `dactyl_generation.FireworksAIClient` to run a batch job with Fireworks AI models and save the results locally.

---

## Prerequisites

```bash
pip install dactyl-generation
```

Create a `.env` file with your Fireworks credentials:

```bash
FIREWORKS_API_KEY=fw-xxxxxxxxxxxxxxxxxxxx
FIREWORKS_ACCOUNT_ID=your_account_id
```

Your input file must be a JSONL file compatible with Fireworks batch prompts, for example (taken from the Fireworks AI documentation):

```json title="fireworks_prompts.jsonl"
{"custom_id": "request-1", "body": {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "max_tokens": 100}}
{"custom_id": "request-2", "body": {"messages": [{"role": "user", "content": "Explain quantum computing"}], "temperature": 0.7}}
{"custom_id": "request-3", "body": {"messages": [{"role": "user", "content": "Tell me a joke"}]}}
```

---

## Creating and Fetching Batch Job Results

Instantiate the client.

```python
import dotenv
dotenv.load_dotenv()

from dactyl_generation.fireworks_generation import FireworksAIClient

client = FireworksAIClient(
    os.environ["FIREWORKS_API_KEY"],
    os.environ["FIREWORKS_ACCOUNT_ID"]
)
```

Define local cache files.

```python
input_file_path = "qwen3-235b-inputs.json"
output_file_path = "qwen3-235b-outputs.json"
```

### Create batch job (first run)

We leave the `inference_parameters` as empty, but feel free to use them. 
```python
results = client.create_batch_job(
    "sample-batch",
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    "fireworks_prompts.jsonl",
    "sample-input-dataset",
    "sample-output-dataset",
    {}
)

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

from dactyl_generation.fireworks_generation import FireworksAIClient

dotenv.load_dotenv()

client = FireworksAIClient(
    os.environ["FIREWORKS_API_KEY"],
    os.environ["FIREWORKS_ACCOUNT_ID"]
)

input_file_path = "qwen3-235b-inputs.json"
output_file_path = "qwen3-235b-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job(
        "sample-batch",
        "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        "fireworks_prompts.jsonl",
        "sample-input-dataset",
        "sample-output-dataset",
        {}
    )
    with open(input_file_path, "w+") as f:
        json.dump(results, f, indent=4)
else:
    client.get_batch_job_output(input_file_path).to_json(
        output_file_path,
        orient="records",
        indent=4
    )
```


