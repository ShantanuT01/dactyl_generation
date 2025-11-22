
# Batch Inference with Google (Gemini)

This short tutorial explains how to use `dactyl_generation.GoogleClient` to run a batch job with Gemini models and save the results locally.

---

## Prerequisites

```bash
pip install dactyl-generation
```

Create a `.env` file with your Gemini API key:

```bash
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxx
```

Your input file should be a JSONL prompt file, for example (taken from Gemini Batch Inference API):

```json title="gemini-sample-prompts.jsonl"
{"key": "request-1", "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}], "generation_config": {"temperature": 0.7}}}
{"key": "request-2", "request": {"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}}
```

---

## Creating and Fetching Batch Job Results

Instantiate the client.

```python
import dotenv
dotenv.load_dotenv()

from dactyl_generation.google_generation import GoogleClient

client = GoogleClient(os.environ["GEMINI_API_KEY"])
```

Define local cache files.

```python
input_file_path = "gemini-2.5-flash-inputs.json"
output_file_path = "gemini-2.5-flash-outputs.json"
```

### Create batch job (first run)

```python
results = client.create_batch_job(
    "gemini-sample-prompts.jsonl",
    "sample-batch",
    "gemini-2.5-flash",
    "test-batch"
)

with open(input_file_path, "w+") as f:
    json.dump(results, f, indent=4)
```

### Fetch results (subsequent runs)

```python
results = client.get_batch_job_output(input_file_path)
results.to_json(
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

from dactyl_generation.google_generation import GoogleClient

dotenv.load_dotenv()

client = GoogleClient(os.environ["GEMINI_API_KEY"])

input_file_path = "gemini-2.5-flash-inputs.json"
output_file_path = "gemini-2.5-flash-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job(
        "gemini-sample-prompts.jsonl",
        "sample-batch",
        "gemini-2.5-flash",
        "test-batch"
    )
    with open(input_file_path, "w+") as f:
        json.dump(results, f, indent=4)
else:
    results = client.get_batch_job_output(input_file_path)
    results.to_json(
        output_file_path,
        orient="records",
        indent=4
    )
```


