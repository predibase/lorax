# Guided Generation

Guided generation (also called constrained decoding) is a technique that forces LLM output to follow certain rules. For 
example, guided generation is useful when you need generated text to parse as valid JSON or even conform to a custom [JSON schema](https://json-schema.org/).

## Background: Guided Generation

During each forward pass of inference, LLMs produce a probability distribution over their vocabulary of tokens. The token 
that is actually generated is selected by sampling from this distribution. 

Suppose you've tasked an LLM with generating some valid JSON, and so far the LLM has produced the text `{ "name"`. When 
considering the next token to output, it's clear that tokens like `A` or `<` will not result in valid JSON. Guided generation
prevents the LLM from selecting an invalid token by modifying the probability distribution and setting the probability of
invalid tokens to something like -infinity. In this way, we can guarantee that, at each step, only tokens that will produce
valid JSON can be selected.

### Important Notes
* Guided generation does not guarantee the _quality_ of generated text, only its _form_. Guided
generation may force the LLM to output valid JSON, but it can't ensure that the content of the JSON is desirable or accurate.
* Even with guided generation enabled, LLM output may not be fully valid JSON if the number of `max_new_tokens` is too low,
    as this could result in necessary tokens (e.g., a closing `}`) being cut off.

## Guided Generation with Outlines

[Outlines](https://github.com/outlines-dev/outlines) is an open-source library supporting various ways of specifying and enforcing
guided generation rules onto LLM outputs.

Currently, LoRAX uses Outlines to support guided generation following a user-provided JSON schema. This JSON schema is
converted into a regular expression, and then into a finite-state machine (FSM). For each token, LoRAX then determines the set of
valid next tokens using this FSM and sets the probability of invalid tokens to -infinity.

## Example

This example follows the [JSON-guided generation example](https://outlines-dev.github.io/outlines/quickstart/#json-guided-generation) in the Outlines quickstart.

We assume that you have already deployed LoRAX using a suitable base model and installed the [LoRAX Python Client](../reference/python_client.md).
Alternatively, see [below](guided_generation.md#openai-compatible-api) for an example of guided generation using an 
OpenAI client.

```python
from lorax import Client

client = Client(endpoint_url)

# Specify your desired format as a JSON schema
schema = {
    "$defs": {
        "Armor": {
            "enum": ["leather", "chainmail", "plate"],
            "title": "Armor",
            "type": "string"
        }
    },
    "properties": {
        "name": {"maxLength": 10, "title": "Name", "type": "string"},
        "age": {"title": "Age", "type": "integer"},
        "armor": {"$ref": "#/$defs/Armor"},
        "strength": {"title": "Strength", "type": "integer"}
    },
    "required": ["name", "age", "armor", "strength"],
    "title": "Character",
    "type": "object"
}

# Now simply pass this schema in the `response_format` parameter when sending a generate request:
prompt = "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength. "
response = client.generate(prompt, response_format={"type": "json_object", "schema": schema})
print(response.generated_text)
```

### OpenAI-compatible API

Guided generation of JSON following a schema is supported via the `response_format` parameter.

NOTE: Currently a schema is REQUIRED. This differs from the existing OpenAI JSON mode, in which no schema is supported.

```python
schema = {
    "$defs": {
        "Armor": {
            "enum": ["leather", "chainmail", "plate"],
            "title": "Armor",
            "type": "string"
        }
    },
    "properties": {
        "name": {"maxLength": 10, "title": "Name", "type": "string"},
        "age": {"title": "Age", "type": "integer"},
        "armor": {"$ref": "#/$defs/Armor"},
        "strength": {"title": "Strength", "type": "integer"}
    },
    "required": ["name", "age", "armor", "strength"],
    "title": "Character",
    "type": "object"
}

# Chat Completions API
resp = client.chat.completions.create(
    model=adapter_id,
    messages=[
        {
            "role": "user",
            "content": "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength. ",
        },
    ],
    max_tokens=100,
    response_format={"type": "json_object", "schema": schema},
)

print("Response:", resp[0].choices[0].message.content)
```


