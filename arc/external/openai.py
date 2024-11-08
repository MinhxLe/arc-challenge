# type: ignore
from typing import Optional, TypeVar, Type
from pydantic.main import BaseModel
from arc import settings
from openai import OpenAI

OAI_MODEL = "gpt-4o-mini"

T = TypeVar("T", bound=BaseModel)


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.OAI_API_KEY)


def complete(msg: str, system_msg: Optional[str] = None) -> str:
    if system_msg is not None:
        msgs = [dict(role="system", content=system_msg)]
    else:
        msgs = []
    msgs.append(dict(role="user", content=msg))

    response = _get_client().chat.completions.create(messages=msgs, model=OAI_MODEL)
    assert len(response.choices) == 1
    return response.choices[0].message.content


def complete_structured(
    msg: str,
    response_format: Type[T],
    system_msg: Optional[str] = None,
) -> T:
    if system_msg is not None:
        msgs = [dict(role="system", content=system_msg)]
    else:
        msgs = []
    msgs.append(dict(role="user", content=msg))
    response = _get_client().beta.chat.completions.parse(
        model=OAI_MODEL, messages=msgs, response_format=response_format
    )

    return response.choices[0].message.parsed
