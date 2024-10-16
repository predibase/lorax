# Structured Output (JSON)

LoRAX can enforce that responses consist only of valid JSON and adhere to a provided [JSON schema](https://json-schema.org/).

## Background: Structured Generation

LoRAX enforces adherence to a schema through a process known as **structured generation** (also called *constrained decoding*). 
Unlike guess-and-check validation methods, structured generation manipulates the next token likelihoods (logits) to enforce adherence to a schema at the token level. During each forward pass of inference, LLMs produce a probability distribution over their vocabulary of tokens. The token 
that is actually generated is selected by sampling from this distribution. 

Suppose you've tasked an LLM with generating some valid JSON, and so far the LLM has produced the text `{ "name"`. When 
considering the next token to output, it's clear that tokens like `A` or `<` will not result in valid JSON. structured generation
prevents the LLM from selecting an invalid token by modifying the probability distribution and setting the likelihood of
invalid tokens to `-infinity`. In this way, we can guarantee that, at each step, only tokens that will produce
valid JSON can be selected.

### Caveats

* Structured generation does not guarantee the _quality_ of generated text, only its _form_. structured
generation may force the LLM to output valid JSON, but it can't ensure that the content of the JSON is desirable or accurate.
* Even with structured generation enabled, LLM output may not be fully valid JSON if the number of `max_new_tokens` is too low,
    as this could result in necessary tokens (e.g., a closing `}`) being cut off.

## Structured Generation with Outlines

[Outlines](https://github.com/outlines-dev/outlines) is an open-source library supporting various ways of specifying and enforcing
structured generation rules onto LLM outputs.

LoRAX uses Outlines to support structured generation following a user-provided JSON schema. This JSON schema is
converted into a regular expression, and then into a finite-state machine (FSM). For each token, LoRAX then determines the set of
valid next tokens using this FSM and sets the likelihood of invalid tokens to `-infinity`.

### Example: Python client

This example follows the [JSON-structured generation example](https://outlines-dev.github.io/outlines/quickstart/#json-structured-generation) in the Outlines quickstart.

We assume that you have already deployed LoRAX using a suitable base model and installed the [LoRAX Python Client](../reference/python_client.md).
Alternatively, see [below](structured_output.md#example-openai-compatible-api) for an example of structured generation using an 
OpenAI client.

```python
import json
from enum import Enum
from lorax import Client
from pydantic import BaseModel, constr


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    strength: int


client = Client("http://127.0.0.1:8080")

# Example 1: Using a schema
prompt_with_schema = "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength."
response_with_schema = client.generate(prompt_with_schema, response_format={
    "type": "json_object",
    "schema": Character.model_json_schema(),
})

my_character_with_schema = json.loads(response_with_schema.generated_text)\
print(my_character_with_schema)
# {
#    "name": "Thorin",
#    "age": 45,
#    "armor": "plate",
#    "strength": 90
# }

# Example 2: Without a schema (arbitrary JSON)
prompt_without_schema = "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength."
response_without_schema = client.generate(prompt_without_schema, response_format={
    "type": "json_object",  # No schema provided
})

my_character_without_schema = json.loads(response_without_schema.generated_text)
print(my_character_without_schema)
# {
#    "characterName": "Aragon",
#    "age": 38,
#    "armorType": "chainmail",
#    "power": 78
# }
```

You can also specify the JSON schema directly rather than using Pydantic:

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
```

### Example: OpenAI-compatible API

Structured generation of JSON following a schema is supported via the `response_format` parameter.

!!! note

    Currently, `response_format` in OpenAI interface differs slightly from the LoRAX request interface.
    When calling the OpenAI-compatible API, you should format the request exactly as specified in the official documentation.
    For more details, refer to the OpenAI documentation here: https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format.

#### Type 1: `text` (default)

- This is the standard mode where the model generates plain text output.
- In this example, the model simply returns plain text output.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8080/v1",
)

resp = client.chat.completions.create(
    model="",  # optional: specify an adapter ID here
    messages=[
        {
            "role": "user",
            "content": "Describe a medieval fantasy character.",
        },
    ],
    max_tokens=100,
    response_format={
        "type": "text",  # Default response type, plain text output
    },
)

print(resp.choices[0].message.content)

'''
Sir Alaric is a noble knight of the realm. At the age of 35, he dons a suit of shining plate armor, protecting his strong, muscular frame. His strength is unparalleled in the kingdom, allowing him to wield his massive greatsword with ease.
'''
```

#### Type 2: `json_object`

- This mode outputs arbitrary JSON objects, making it ideal for generating data in a flexible JSON format without enforcing any schema. It's similar to OpenAI’s JSON mode.
- In this example, the model returns an arbitrary JSON object without enforcing a predefined schema.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8080/v1",
)

resp = client.chat.completions.create(
    model="",  # optional: specify an adapter ID here
    messages=[
        {
            "role": "user",
            "content": "Generate a new character for my game: name, age, armor type, and strength.",
        },
    ],
    max_tokens=100,
    response_format={
        "type": "json_object",  # Generate arbitrary JSON without a schema
    },
)

my_character = json.loads(resp.choices[0].message.content)
print(my_character)

'''
{
    "name": "Eldrin",
    "age": 27,
    "armor": "Dragonscale Armor",
    "strength": "Fire Resistance"
}
'''
```

#### Type 3: `json_schema`

- The model returns a structured JSON object that adheres to the predefined schema. This ensures that the JSON follows the format of the `Character` model provided earlier.
- In this example, the model generates structured JSON output that adheres to a predefined schema.

```python
import json
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel, constr


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    strength: int


client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8080/v1",
)

resp = client.chat.completions.create(
    model="",  # optional: specify an adapter ID here
    messages=[
        {
            "role": "user",
            "content": "Generate a new character for my game: name, age (between 1 and 99), armor, and strength.",
        },
    ],
    max_tokens=100,
    response_format={
        "type": "json_schema",  # Generate structured JSON output based on a schema
        "json_schema": {
            "name": "Character",  # Name of the schema
            "schema": Character.model_json_schema(),  # The JSON schema generated by Pydantic
        },
    },
)

my_character = json.loads(resp.choices[0].message.content)
print(my_character)

'''
{
    "name": "Thorin",
    "age": 45,
    "armor": "plate",
    "strength": 90
}
'''
```


