from dataclasses import dataclass
from typing import Callable
from arc.core import Grid


@dataclass
class Program:
    source: str
    fn: Callable[[Grid], Grid]

    @classmethod
    def from_source(cls, source: str) -> "Program":
        local = dict()
        exec(source, local)
        assert "transform" in local
        fn = local["transform"]
        return Program(source, fn)
