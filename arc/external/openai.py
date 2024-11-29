# type: ignore
from typing import Optional, TypeVar, Type
from pydantic.main import BaseModel
from arc import settings
from openai import OpenAI

OAI_MODEL = "gpt-4o"

T = TypeVar("T", bound=BaseModel)


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.OAI_API_KEY)


def complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    return_raw: bool = False,
) -> str:
    if system_prompt is not None:
        prompts = [dict(role="system", content=system_prompt)]
    else:
        prompts = []
    prompts.append(dict(role="user", content=prompt))

    response = _get_client().chat.completions.create(
        messages=prompts,
        model=OAI_MODEL,
        temperature=temperature,
    )
    assert len(response.choices) == 1
    if not return_raw:
        return response.choices[0].message.content
    else:
        return response


def complete_structured(
    prompt: str,
    response_format: Type[T],
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    return_raw: bool = False,
) -> T:
    if system_prompt is not None:
        prompts = [dict(role="system", content=system_prompt)]
    else:
        prompts = []
    prompts.append(dict(role="user", content=prompt))
    response = _get_client().beta.chat.completions.parse(
        model=OAI_MODEL,
        messages=prompts,
        response_format=response_format,
        temperature=temperature,
    )
    if not return_raw:
        return response_format.model_validate_json(response.choices[0].message.content)
    else:
        return response
