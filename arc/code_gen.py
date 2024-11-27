import re
from typing import Optional, Callable
from loguru import logger


def parse_code(response: str) -> Optional[str]:
    # Extract code between Python code blocks
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)

    if code_match is None:
        return None
    else:
        return (
            code_match.group(1)
            .strip()
            .replace("from common import", "from arc.dsl.common import")
        )


def interpret_transform_fn(source_code: str) -> Callable | None:
    try:
        local = dict()
        exec(source_code, local)
        assert "transform" in local
        return local["transform"]
    except Exception:
        logger.exception("failed to interpret code")
        return None
