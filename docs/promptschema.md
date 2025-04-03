# Prompt Schema

We define a few-shot prompt schema for an individual prompt as follows:

```json
{
  "title": "Few Shot Prompt",
  "description": "A few shot prompt to pass to the dactyl_generation library.",
  "type": "object",
  "properties": {
    "system_prompt": {
      "description": "The system prompt to pass to the LLM.",
      "type": "string"
    },
    "examples": {
      "description": "List of examples (strings) to pass to for this system prompt.",
      "type": "array"
    }
  }
}
```

An example of a system prompt with one-shot examples would look like:

```json
{
  "system_prompt": "System prompt goes here.",
  "examples": ["Example 1"]
}
```