from __future__ import annotations

import ast
import hashlib
import io
import os
import subprocess
import tokenize
import warnings
from pathlib import Path

from .model import Inventory, Kernel, Repository, ScanIssue

DEFAULT_EXCLUDES = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "__pycache__", "build", "dist", "node_modules",
}


def _git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True
    )
    if check and result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"{path} is not a usable Git repository: {message}")
    return result.stdout.strip()


def repository_info(path: str | Path) -> Repository:
    requested = Path(path).expanduser().resolve()
    if not requested.is_dir():
        raise ValueError(f"repository path is not a directory: {requested}")
    git_root = Path(_git(requested, "rev-parse", "--show-toplevel")).resolve()
    commit = _git(git_root, "rev-parse", "HEAD")
    branch = _git(git_root, "symbolic-ref", "--short", "HEAD", check=False) or "DETACHED"
    revision = _git(git_root, "describe", "--always", "--tags", "--dirty", check=False)
    if not revision:
        revision = commit[:12]
    dirty = bool(_git(git_root, "status", "--porcelain", check=False))
    return Repository(
        path=str(requested), git_root=str(git_root), revision=revision,
        commit=commit, branch=branch, dirty=dirty,
    )


def _python_files(root: Path, excludes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in excludes)
        for filename in sorted(files):
            if filename.endswith(".py"):
                paths.append(Path(directory) / filename)
    return paths


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _triton_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    module_aliases: set[str] = set()
    jit_aliases: set[str] = set()
    # Imports are sometimes guarded by try/except for optional backends.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton":
                    module_aliases.add(alias.asname or "triton")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module == "triton" and alias.name == "jit":
                    jit_aliases.add(alias.asname or "jit")
                # vLLM intentionally re-exports the optional Triton module so
                # importing vLLM itself remains possible without Triton.
                elif node.module == "vllm.triton_utils" and alias.name == "triton":
                    module_aliases.add(alias.asname or "triton")
    return module_aliases, jit_aliases


def _is_triton_jit(node: ast.AST, modules: set[str], jits: set[str]) -> bool:
    name = _dotted_name(node)
    if not name:
        return False
    if name in jits:
        return True
    parts = name.split(".")
    return len(parts) == 2 and parts[0] in modules and parts[1] == "jit"


def _source_segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    return segment.strip() if segment else ""


def _semantic_dump(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    # Exclude the function name, positions, comments, and formatting. Keep the
    # complete signature, decorator configuration, return annotation, and body.
    parts = [
        ast.dump(node.args, annotate_fields=True, include_attributes=False),
        "|".join(ast.dump(d, include_attributes=False) for d in node.decorator_list),
        ast.dump(node.returns, include_attributes=False) if node.returns else "",
        "|".join(ast.dump(item, include_attributes=False) for item in node.body),
    ]
    return "\n".join(parts)


def _tokens(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    # ast.unparse makes matching insensitive to comments and layout. Function
    # names are deliberately omitted so exact code can survive a rename.
    text = "\n".join(ast.unparse(item) for item in node.body)
    result: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in stream:
            if token.type in {tokenize.NAME, tokenize.NUMBER, tokenize.OP, tokenize.STRING}:
                result.append(token.string)
    except (IndentationError, tokenize.TokenError):
        pass
    return tuple(result)


def _signature(text: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    line = text.splitlines()[node.lineno - 1]
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    try:
        rendered = ast.unparse(node.args)
    except Exception:
        rendered = line.strip()
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix}{node.name}({rendered}){returns}"


class _KernelVisitor(ast.NodeVisitor):
    def __init__(self, text: str, relative_path: str, modules: set[str], jits: set[str]):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.relative_path = relative_path
        self.modules = modules
        self.jits = jits
        self.parents: list[str] = []
        self.kernels: list[Kernel] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.parents.append(node.name)
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.parents, node.name])
        jit_decorators = [d for d in node.decorator_list if _is_triton_jit(d, self.modules, self.jits)]
        if jit_decorators:
            start = min(d.lineno for d in node.decorator_list)
            end = node.end_lineno or node.lineno
            source = "".join(self.lines[start - 1:end]).rstrip() + "\n"
            semantic = _semantic_dump(node)
            identity = f"{self.relative_path}:{qualname}:{start}"
            self.kernels.append(Kernel(
                id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                path=self.relative_path,
                module=self.relative_path[:-3].replace("/", "."),
                name=node.name,
                qualname=qualname,
                decorator_line=start,
                definition_line=node.lineno,
                end_line=end,
                decorators=tuple(_source_segment(self.text, d) for d in node.decorator_list),
                signature=_signature(self.text, node),
                source=source,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                semantic_sha256=hashlib.sha256(semantic.encode()).hexdigest(),
                body_tokens=_tokens(node),
                syntax_kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            ))
        self.parents.append(node.name)
        self.generic_visit(node)
        self.parents.pop()


def scan_repository(path: str | Path, excludes: set[str] | None = None) -> Inventory:
    repository = repository_info(path)
    root = Path(repository.path)
    excluded = DEFAULT_EXCLUDES | (excludes or set())
    kernels: list[Kernel] = []
    issues: list[ScanIssue] = []
    files = _python_files(root, excluded)
    for source_path in files:
        relative = source_path.relative_to(root).as_posix()
        try:
            text = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            issues.append(ScanIssue(relative, f"cannot read file: {exc}"))
            continue
        try:
            with warnings.catch_warnings():
                # A few repository scripts contain legacy string escapes. They
                # are valid AST input and unrelated to Triton kernel discovery.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text, filename=relative, type_comments=True)
        except SyntaxError as exc:
            issues.append(ScanIssue(relative, exc.msg, exc.lineno))
            continue
        modules, jits = _triton_aliases(tree)
        visitor = _KernelVisitor(text, relative, modules, jits)
        visitor.visit(tree)
        kernels.extend(visitor.kernels)
    kernels.sort(key=lambda item: (item.path, item.decorator_line, item.qualname))
    return Inventory(repository, kernels, len(files), issues)
