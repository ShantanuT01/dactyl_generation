
# Batch Inference with Mistral

This short tutorial explains how to use `dactyl_generation.MistralClient` to run a batch job with Mistral models and save the results locally.

---

## Prerequisites

```bash
pip install dactyl-generation
```

Create a `.env` file with your Mistral API key:

```bash
MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxx
```

Your `local-tiny-test.json` should contain prompt data, for example:

```json title="local-tiny-test.json"
 [
  {
    "custom_id":"custom_id_0",
    "messages":[
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
    "top_p":0.8501584248,
    "model": "mistral-small-latest",
    "frequency_penalty": 1.1
    }
]
```

---

## Creating and Fetching Batch Job Results

We first load the prompt dataset.

```python
import dotenv
import pandas as pd

dotenv.load_dotenv()
prompts_df = pd.read_json("local-tiny-test.json")
```

Instantiate the client.

```python
from dactyl_generation.mistral_generation import MistralClient

client = MistralClient(os.environ["MISTRAL_API_KEY"])
```

Define local cache files.

```python
input_file_path = "mistral-small-latest-inputs.json"
output_file_path = "mistral-small-latest-outputs.json"
```

### Create batch job (first run)

```python
results = client.create_batch_job("sample", prompts_df)
with open(input_file_path, "w+") as f:
    json.dump(results, f, indent=4)
```

### Fetch results (subsequent runs)

```python
results = client.get_batch_job_output(input_file_path)
results.to_json(
    output_file_path,
    index=False,
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

from dactyl_generation.mistral_generation import MistralClient

dotenv.load_dotenv()
prompts_df = pd.read_json("local-tiny-test.json")

client = MistralClient(os.environ["MISTRAL_API_KEY"])

input_file_path = "mistral-small-latest-inputs.json"
output_file_path = "mistral-small-latest-outputs.json"

if not os.path.exists(input_file_path):
    results = client.create_batch_job("sample", prompts_df)
    with open(input_file_path, "w+") as f:
        json.dump(results, f, indent=4)
else:
    results = client.get_batch_job_output(input_file_path)
    results.to_json(
        output_file_path,
        index=False,
        orient="records",
        indent=4
    )
```


