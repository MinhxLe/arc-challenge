# type: ignore
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, Optional, TypeVar, Type
from openai.types.chat import ChatCompletion
from pydantic.main import BaseModel
from arc import settings
from openai import OpenAI, AsyncOpenAI

OAI_MODEL = "gpt-4o-mini-2024-07-18"


# pricing by token
@dataclass
class Pricing:
    prompt: Decimal
    cached_prompt: Decimal
    completion: Decimal

    batch_prompt: Decimal
    batch_completion: Decimal


PRICING_BY_MODEL = {
    "gpt-4o-mini-2024-07-18": Pricing(
        prompt=0.15,
        cached_prompt=0.075,
        completion=0.6,
        batch_prompt=0.075,
        batch_completion=0.3,
    )
}

T = TypeVar("T", bound=BaseModel)


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.OAI_API_KEY)


def _get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OAI_API_KEY)


@dataclass
class RawCompletion(Generic[T]):
    completion: T
    response: ChatCompletion


def complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    return_raw: bool = False,
) -> str | RawCompletion:
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
    completion = response.choices[0].message.content
    if not return_raw:
        return completion
    else:
        return RawCompletion(completion, response)


def complete_structured(
    prompt: str,
    response_format: Type[T],
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    return_raw: bool = False,
) -> T | RawCompletion:
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
    completion = response_format.model_validate_json(
        response.choices[0].message.content
    )
    if not return_raw:
        completion
    else:
        return RawCompletion(completion, response)


async def complete_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    return_raw: bool = False,
) -> str | RawCompletion:
    if system_prompt is not None:
        prompts = [dict(role="system", content=system_prompt)]
    else:
        prompts = []
    prompts.append(dict(role="user", content=prompt))

    response = await _get_async_client().chat.completions.create(
        messages=prompts,
        model=OAI_MODEL,
        temperature=temperature,
    )
    assert len(response.choices) == 1
    completion = response.choices[0].message.content
    if not return_raw:
        return completion
    else:
        return RawCompletion(completion, response)


async def complete_structured_async(
    prompt: str,
    response_format: Type[T],
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    return_raw: bool = False,
) -> T | RawCompletion:
    if system_prompt is not None:
        prompts = [dict(role="system", content=system_prompt)]
    else:
        prompts = []
    prompts.append(dict(role="user", content=prompt))
    response = await _get_async_client().beta.chat.completions.parse(
        model=OAI_MODEL,
        messages=prompts,
        response_format=response_format,
        temperature=temperature,
    )
    completion = response_format.model_validate_json(
        response.choices[0].message.content
    )
    if not return_raw:
        completion
    else:
        return RawCompletion(completion, response)


def calculate_cost(response: ChatCompletion) -> Decimal:
    pricing = PRICING_BY_MODEL[response.model]
    usage = response.usage
    cached_prompt_tokens = usage.prompt_tokens_details.cached_tokens
    prompt_tokens = usage.prompt_tokens - cached_prompt_tokens
    completion_tokens = usage.completion_tokens
    prompt_cost = prompt_tokens * pricing.prompt / 1_000_000
    cached_prompt_cost = cached_prompt_tokens * pricing.cached_prompt / 1_000_000
    completion_cost = completion_tokens * pricing.completion / 1_000_000
    return prompt_cost + cached_prompt_cost + completion_cost
