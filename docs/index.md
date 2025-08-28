
# ![pterosaur flying icon](assets/icons8-pterodactyl-48.svg) DACTYL Generation

## About

DACTYL (Diverse Adversarial Corpus of Texts Yielded from Large language models) is an AI-generated (AIG) text detection dataset created for my MPhil thesis project. 

We aim to build a dataset with the following properties:

- :thinking: Challenging AIG texts: Most AIG text datasets contain trivial examples. 
- :material-elevation-rise: Adaptable: We aim for living dataset that grows as AIG text detection evolves.
- :material-diversify: Diverse: We support six domains vulnerable to LLM abuse.

As part of this project, a helper library was developed to support text generation from different APIs &mdash; `dactyl-generation`.
This library can be used to submit batch responses rapidly for various providers. 

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
