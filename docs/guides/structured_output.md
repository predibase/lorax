# Structured Output (JSON)

LoRAX can enforce that responses consist only of valid JSON and adhere to a provided [JSON schema](https://json-schema.org/).

## Background: Guided Generation

LoRAX enforces adherence to a schema through a process known as **guided generation** (also called *constrained decoding*). 
Unlike guess-and-check validation methods, guided generation manipulates the next token likelihoods (logits) to enforce adherence to a schema at the token level. During each forward pass of inference, LLMs produce a probability distribution over their vocabulary of tokens. The token 
that is actually generated is selected by sampling from this distribution. 

Suppose you've tasked an LLM with generating some valid JSON, and so far the LLM has produced the text `{ "name"`. When 
considering the next token to output, it's clear that tokens like `A` or `<` will not result in valid JSON. Guided generation
prevents the LLM from selecting an invalid token by modifying the probability distribution and setting the likelihood of
invalid tokens to `-infinity`. In this way, we can guarantee that, at each step, only tokens that will produce
valid JSON can be selected.

### Caveats

* Guided generation does not guarantee the _quality_ of generated text, only its _form_. Guided
generation may force the LLM to output valid JSON, but it can't ensure that the content of the JSON is desirable or accurate.
* Even with guided generation enabled, LLM output may not be fully valid JSON if the number of `max_new_tokens` is too low,
    as this could result in necessary tokens (e.g., a closing `}`) being cut off.

## Guided Generation with Outlines

[Outlines](https://github.com/outlines-dev/outlines) is an open-source library supporting various ways of specifying and enforcing
guided generation rules onto LLM outputs.

LoRAX uses Outlines to support guided generation following a user-provided JSON schema. This JSON schema is
converted into a regular expression, and then into a finite-state machine (FSM). For each token, LoRAX then determines the set of
valid next tokens using this FSM and sets the likelihood of invalid tokens to `-infinity`.

### Example: Python client

This example follows the [JSON-guided generation example](https://outlines-dev.github.io/outlines/quickstart/#json-guided-generation) in the Outlines quickstart.

We assume that you have already deployed LoRAX using a suitable base model and installed the [LoRAX Python Client](../reference/python_client.md).
Alternatively, see [below](structured_output.md#openai-compatible-api) for an example of guided generation using an 
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

prompt = "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength. "
response = client.generate(prompt, response_format={
    "type": "json_object",
    "schema": Character.model_json_schema(),
})

my_character = json.loads(response.generated_text)
print(my_character)
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

Guided generation of JSON following a schema is supported via the `response_format` parameter.

!!! note

    Currently a schema is **required**. This differs from the existing OpenAI JSON mode, in which no schema is supported.

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
            "content": "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength. ",
        },
    ],
    max_tokens=100,
    response_format={
        "type": "json_object",
        "schema": Character.model_json_schema(),
    },
)

my_character = json.loads(resp[0].choices[0].message.content)
print(my_character)
```


