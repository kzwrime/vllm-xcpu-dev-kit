import os
import shutil
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.cwd() / "torch_compile_cache"))

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config


torch.set_num_threads(1)
inductor_config.cpp_wrapper = True
inductor_config.compile_threads = 1


def set_case_cache(name):
    cache = Path.cwd() / "torch_compile_cache" / name
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    dynamo.reset()
    return cache


def cache_files():
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    return sorted(p for p in cache.rglob("*") if p.is_file())


def print_interesting_files(title):
    print(f"\n=== {title}: generated files ===")
    for p in cache_files():
        rel = p.relative_to(Path.cwd())
        if p.suffix in {".cpp", ".cc", ".py"} or "wrapper" in p.name:
            print(rel)


def grep_generated(patterns):
    print("\n=== generated code snippets ===")
    for p in cache_files():
        if p.suffix not in {".cpp", ".cc", ".py"}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except UnicodeDecodeError:
            continue
        hits = []
        for pat in patterns:
            idx = text.find(pat)
            if idx >= 0:
                start = max(0, idx - 240)
                end = min(len(text), idx + 500)
                hits.append((pat, text[start:end]))
        if hits:
            print(f"\n--- {p.relative_to(Path.cwd())} ---")
            for pat, snippet in hits:
                print(f"[hit: {pat!r}]")
                print(snippet)


class BoolBranch(torch.nn.Module):
    def forward(self, x, is_decode: bool):
        if is_decode:
            return x + 1.0
        return x * 2.0


class TensorCond(torch.nn.Module):
    def forward(self, x, is_decode):
        return torch.cond(
            is_decode,
            lambda t: t + 1.0,
            lambda t: t * 2.0,
            [x],
        )


def run_bool_branch():
    set_case_cache("python_bool")
    print("\n\n######## Python bool branch ########")
    mod = torch.compile(BoolBranch(), backend="inductor", fullgraph=True, dynamic=True)
    x = torch.arange(4, dtype=torch.float32)
    print("true result ", mod(x, True))
    print("false result", mod(x, False))
    print("true again  ", mod(x, True))
    print_interesting_files("python bool")
    grep_generated(["+ 1.0", "* 2.0", "if (", "if("])


def run_tensor_cond():
    set_case_cache("torch_cond_tensor")
    print("\n\n######## torch.cond tensor predicate ########")
    mod = torch.compile(TensorCond(), backend="inductor", fullgraph=True, dynamic=True)
    x = torch.arange(4, dtype=torch.float32)
    pred_true = torch.tensor(True)
    pred_false = torch.tensor(False)
    print("true result ", mod(x, pred_true))
    print("false result", mod(x, pred_false))
    print("true again  ", mod(x, pred_true))
    print_interesting_files("torch.cond tensor")
    grep_generated(["cond", "if (", "if(", "+ 1.0", "* 2.0"])


def run_tensor_item_branch():
    set_case_cache("tensor_item_python_if")
    print("\n\n######## Python if on tensor.item() ########")

    class TensorItemBranch(torch.nn.Module):
        def forward(self, x, is_decode):
            if is_decode.item():
                return x + 1.0
            return x * 2.0

    mod = torch.compile(TensorItemBranch(), backend="inductor", fullgraph=True, dynamic=True)
    x = torch.arange(4, dtype=torch.float32)
    for pred in (torch.tensor(True), torch.tensor(False)):
        try:
            print("pred", pred, "=>", mod(x, pred))
        except Exception as exc:
            print(type(exc).__name__, exc)
            break
    print_interesting_files("tensor.item branch")
    grep_generated(["+ 1.0", "* 2.0", "if (", "if("])


if __name__ == "__main__":
    print("torch", torch.__version__)
    print("cpp_wrapper", inductor_config.cpp_wrapper)
    print("cache", os.environ["TORCHINDUCTOR_CACHE_DIR"])
    run_bool_branch()
    run_tensor_cond()
    run_tensor_item_branch()
