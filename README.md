
# ![pterosaur flying icon](docs/assets/icons8-pterodactyl-48.svg) DACTYL Generation
![PyPI](https://img.shields.io/pypi/v/dactyl-generation?label=pypi%20package) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/dactyl-generation?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BRIGHTGREEN&left_text=downloads)](https://pepy.tech/projects/dactyl-generation) [![arXiv](https://img.shields.io/badge/arXiv-2508.00619-b31b1b.svg)](https://arxiv.org/abs/2508.00619) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/ShantanuT01/DACTYL)
## About

DACTYL (Diverse Adversarial Corpus of Texts Yielded from Large language models) is an AI-generated (AIG) text detection dataset created for my MPhil thesis project. 

We aim to build a dataset with the following properties:

- :thinking: Challenging AIG texts: Most AIG text datasets contain trivial examples. 
- :material-elevation-rise: Adaptable: We aim for living dataset that grows as AIG text detection evolves.
- :material-diversify: Diverse: We support six domains vulnerable to LLM abuse.

As part of this project, a helper library was developed to support text generation from different APIs &mdash; `dactyl-generation`.
This library can be used to submit batch responses rapidly for various providers.

## Installation and Setup

```bash
pip install dactyl-generation
```

Make sure to have an `.env` file containing the following values (the API keys can be empty if you do not have them).

```text
OPENAI_API_KEY="your-api-key-goes-here"
GEMINI_API_KEY=""
MISTRAL_API_KEY=""
ANTHROPIC_API_KEY=""
FIREWORKS_AI_API_KEY=""
```

Check the `examples/` directories to see how to use different batch inference providers seamlessly. 


## Supported Providers

We support batch inference for some providers. Batch inference often gives a 50% discount compared to using the streaming API. 

|   **Provider**   | **Batch** |   **Streaming**    |
|:----------------:|:---------:|:------------------:|
|    **OpenAI**    |  :white_check_mark:         |        :x:         |
|   **Anthropic**  |   :white_check_mark:        |        :x:         |
|    **Mistral**   |   :white_check_mark:        |        :x:         |
|    **Gemini**    |     :x:      | :white_check_mark: |
| **Fireworks AI** |    :x:       | :white_check_mark: |
|    **Bedrock**   |     :white_check_mark:      | :white_check_mark: |


Icons by [Icons8](https://icons8.com/). 
