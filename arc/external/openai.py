# type: ignore
from arc import settings
from openai import OpenAI

OAI_MODEL = "gpt-4o-mini"


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.OAI_API_KEY)


def complete(message: str) -> str:
    response = _get_client().chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model=OAI_MODEL,
    )
    assert len(response.choices) == 1
    return response.choices[0].message.content
