"""
import os
import json
import openai

client = openai.OpenAI(
    base_url = "https://api.together.xyz/v1",
    api_key = os.environ['TOGETHER_API_KEY'],
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "celsius",
                            "fahrenheit"
                        ]
                    }
                }
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls."},
    {"role": "user", "content": "What is the current temperature of New York, San Francisco and Chicago?"}
]

response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(json.dumps(response.choices[0].message.model_dump()['tool_calls'], indent=2))

[
  {
    "id": "call_1p75qwks0etzfy1g6noxvsgs",
    "function": {
      "arguments": "{\"location\":\"New York, NY\",\"unit\":\"fahrenheit\"}",
      "name": "get_current_weather"
    },
    "type": "function"
  },
  {
    "id": "call_aqjfgn65d0c280fjd3pbzpc6",
    "function": {
      "arguments": "{\"location\":\"San Francisco, CA\",\"unit\":\"fahrenheit\"}",
      "name": "get_current_weather"
    },
    "type": "function"
  },
  {
    "id": "call_rsg8muko8hymb4brkycu3dm5",
    "function": {
      "arguments": "{\"location\":\"Chicago, IL\",\"unit\":\"fahrenheit\"}",
      "name": "get_current_weather"
    },
    "type": "function"
  }


"""
import json
from typing import Any, Annotated

from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator
import jsonschema

_DEFAULT_TOOLS_PROMPT_TEMPLATE = ("You are a helpful assistant that can access external functions. "
                                  "The responses from these function calls will be appended to this dialogue. "
                                  "Please provide responses based on the information from these function calls. "
                                  "The functions available to you are as follows:\n%s\n")


def _check_valid_jsonschema(x: dict[str, Any]) -> dict[str, Any]:
    jsonschema.Draft202012Validator.check_schema(x)
    return x


class FunctionDefinition(BaseModel):
    name: str
    description: str = Field(default="")
    parameters: Annotated[dict[str, Any], AfterValidator(_check_valid_jsonschema)]


def tools_prompt(*args: FunctionDefinition) -> str:
    return _DEFAULT_TOOLS_PROMPT_TEMPLATE % "\n".join(json.dumps(fd) for fd in args)


def schema(fns: list[FunctionDefinition]) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "oneOf": [f.parameters for f in fns]
        }
    }
