from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Repository:
    path: str
    git_root: str
    revision: str
    commit: str
    branch: str
    dirty: bool


@dataclass(frozen=True)
class Kernel:
    id: str
    path: str
    module: str
    name: str
    qualname: str
    decorator_line: int
    definition_line: int
    end_line: int
    decorators: tuple[str, ...]
    signature: str
    source: str
    source_sha256: str
    semantic_sha256: str
    body_tokens: tuple[str, ...]
    syntax_kind: str = "function"

    @property
    def line_count(self) -> int:
        return self.end_line - self.decorator_line + 1


@dataclass(frozen=True)
class ScanIssue:
    path: str
    message: str
    line: int | None = None


@dataclass
class Inventory:
    repository: Repository
    kernels: list[Kernel]
    scanned_python_files: int
    issues: list[ScanIssue] = field(default_factory=list)


@dataclass(frozen=True)
class DiffStats:
    added: int
    deleted: int
    changed: int


@dataclass
class Match:
    old: Kernel | None
    new: Kernel | None
    status: str
    method: str
    confidence: str
    score: float
    stats: DiffStats
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    diff_file: str | None = None

    @property
    def changed(self) -> bool:
        return self.status != "unchanged"


@dataclass
class Comparison:
    old: Inventory
    new: Inventory
    matches: list[Match]


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value

