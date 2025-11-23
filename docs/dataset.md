# DACTYL Dataset

[🤗 HuggingFace dataset](https://huggingface.co/datasets/ShantanuT01/DACTYL)

The Diverse Adversarial Corpus of Texts Yielded from Large language models was the first dataset created by the `dactyl-generation` library. 

The DACTYL dataset is an AI-generated text detection dataset focusing primarily on one-shot or few-shot examples. We also include texts from continued pre-trained small language models. 

For more information, refer to our [paper](https://arxiv.org/abs/2508.00619). 

## Models Used

We used the following LLMs to generate texts.
- OpenAI’s GPT-4o-mini and GPT-4o 
- Anthropic’s Claude Haiku and Sonnet 3.5 
- Mistral Small (24B)and Large 2 (123B) 
- Google’s Gemini 1.5 Flash and Pro
- Meta’s Llama 3.2 90B and 3.3 70B 
- DeepSeek-V3 (671B)

We trained Llama-3.2 1B models for generating additional texts. The `dactyl-generation` package was used to generate texts.

## Domains

### Non-adversarial

We define non-adversarial texts as texts generated from the 11 LLMs.

| Domain         | Training (Human) | Training (AI) | Validation (Human) | Validation (AI) | Testing (Human) | Testing (AI) | Total   |
|----------------|:----------------:|:-------------:|:------------------:|:---------------:|:---------------:|:------------:|:-------:|
| Tweets         |      56801       |     16500     |        7066        |      6600       |      7080       |     6600     | 100647  |
| Reviews        |      68000       |     11000     |       17000        |      2750       |     17000       |     2750     | 118500  |
| Abstracts      |      80000       |     33000     |       10000        |     11000       |     11000       |    11000     | 155000  |
| News           |      35916       |     10560     |        4489        |      3520       |      4493       |     3520     | 62498   |
| Student Essays |      83128       |     7920      |       10783        |      4268       |     12571       |     4268     | 122938  |
| Writing Prompts|      50000       |     5500      |       10000        |      2200       |     10000       |     2200     | 79900   |
| **Total**      |   **373845**     |   **84480**   |      **59338**     |    **30338**    |    **61144**    |   **30338**  | **639483**   |

### Adversarial

Adversarial texts refer to the continued pre-training (CPT) generations.

| Domain          | Training (Base) | Training (CPT) | Validation (Base) | Validation (CPT) | Testing (Base) | Testing (CPT) | Total  |
|-----------------|:---------------:|:--------------:|:-----------------:|:----------------:|:--------------:|:-------------:|:------:|
| Tweets          |        0        |      1500      |         0         |       600        |      600       |      600      |  3300  |
| Reviews         |        0        |      1000      |         0         |       250        |      250       |      250      |  1750  |
| Abstracts       |        0        |      3000      |         0         |      1000        |     1000       |     1000      |  6000  |
| News            |        0        |       960      |         0         |       320        |      320       |      320      |  1920  |
| Student Essays  |        0        |       720      |         0         |       388        |      388       |      388      |  1884  |
| Creative Writing|        0        |       500      |         0         |       200        |      200       |      200      |  1100  |
| **Total**       |     **0**       |    **7680**    |      **0**        |     **2758**     |    **2758**    |    **2758**   | **15954** |



## Citation

```bibtex
@misc{thorat2025dactyldiverseadversarialcorpus,
      title={DACTYL: Diverse Adversarial Corpus of Texts Yielded from Large Language Models}, 
      author={Shantanu Thorat and Andrew Caines},
      year={2025},
      eprint={2508.00619},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.00619}, 
}
```