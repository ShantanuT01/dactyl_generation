
# ![pterosaur flying icon](assets/icons8-pterodactyl-48.svg) DACTYL Generation

## About

DACTYL (Diverse Adversarial Corpus of Texts Yielded from Large language models) is an AI-generated (AIG) text detection dataset created for my MPhil thesis project. 

We aim to build a dataset with the following properties:

- :thinking: Challenging AIG texts: Most AIG text datasets contain trivial examples. 
- :material-elevation-rise: Adaptable: We aim for living dataset that grows as AIG text detection evolves.
- :material-diversify: Diverse: We are currently at six domains, but we intend to expand to more.

## Supported Models

We support the following batch APIs which allow for significant cost savings.

| Batch Model       | Model ID                     |
|-------------------|------------------------------|
| GPT-4o-mini       | `gpt-4o-mini`                |
| GPT-4o            | `gpt-4o-2024-11-20`          |
| Claude 3.5 Haiku  | `claude-3-5-haiku-20241022`  |
| Claude 3.5 Sonnet | `claude-3-5-sonnet-20241022` |
| Mistral Small     | `mistral-small-latest`       |
| Mistral Large     | `mistral-large-latest`       |

We also support the following streaming models.

| Streaming Model                | Model Entry                                                      |
|--------------------------------|------------------------------------------------------------------|
| Gemini 1.5 Flash               | `gemini-1.5-flash`                                                 |
| Gemini 1.5 Pro                 | `gemini-1.5-pro`                                                   |
| Bedrock Llama 3.3 70B Instruct | `bedrock/us.meta.llama3-3-70b-instruct-v1:0`                       |
| Bedrock Llama 3.2 90B Instruct | `bedrock/us.meta.llama3-2-90b-instruct-v1:0`                       |
| DeepSeek-V3                    | `deepseek-ai/DeepSeek-V3`                                          |
