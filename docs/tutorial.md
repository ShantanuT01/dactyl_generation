# Tutorial

## Installation

```commandline
pip install dactyl-generation
```

Next include a `.env` file with your API keys. If you don't have an API key, feel free to leave the corresponding field blank.

```text
OPENAI_API_KEY="your-api-key-goes-here"
GEMINI_API_KEY=""
MISTRAL_API_KEY=""
ANTHROPIC_API_KEY=""
FIREWORKS_AI_API_KEY=""
```
!!! note 
    AWS Bedrock (`boto3`) requires a different setup. Make sure that you have boto3 installed in the same environment as `dactyl_generation`!


## Using batch inference for GPT-5

Store your desired prompts in a JSON file. For example:

**openai-local-tiny-test.json**
```json
 [
  {
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
```
```python
import dotenv
dotenv.load_dotenv()

from dactyl_generation.quick import generate_texts_using_batch
import pandas as pd

df = pd.read_json("openai-local-tiny-test.json")

generate_texts_using_batch("gpt-5", "openai-test-prompts.json",df, 512)
```

After the batch runs, you can fetch the response results:
```python
from dactyl_generation.quick import get_batch_job_results
get_batch_job_results("openai-test-prompts.json","openai-batch-results.json")
```
