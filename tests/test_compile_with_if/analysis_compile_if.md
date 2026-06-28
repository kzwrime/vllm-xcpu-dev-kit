# PyTorch compile + cpp_wrapper branch experiment

Environment:

- `torch 2.10.0a0+git2395fec`
- CPU backend
- `torch._inductor.config.cpp_wrapper = True`

## Conclusion

`torch.compile` cannot turn an ordinary Python `if is_decode:` with a Python
`bool` argument into a runtime branch in the generated C++ wrapper. Dynamo
specializes on the Python bool value, guards it, and compiles separate graphs.

If a real generated C++ `if` is required, use `torch.cond` with a 0-d bool
tensor predicate:

```python
def forward(self, x, is_decode):
    return torch.cond(
        is_decode,
        lambda t: t + 1.0,
        lambda t: t * 2.0,
        [x],
    )
```

Then pass `torch.tensor(True)` / `torch.tensor(False)` instead of a Python bool.
In this environment Inductor generated a single C++ wrapper containing:

```cpp
AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_bool(arg0_1, &arg0_1_scalar));
if (arg0_1_scalar) {
    // subgraph: true_graph_0
    ...
} else {
    // subgraph: false_graph_0
    ...
}
```

## Evidence

Run:

```bash
python experiment_compile_if.py
```

For a Python bool branch:

```python
class BoolBranch(torch.nn.Module):
    def forward(self, x, is_decode: bool):
        if is_decode:
            return x + 1.0
        return x * 2.0
```

The calls `mod(x, True)`, `mod(x, False)`, `mod(x, True)` produced correct
results, but generated two separate wrapper files:

- `torch_compile_cache/python_bool/a6/...main.cpp` contains only the `+ 1.0`
  branch.
- `torch_compile_cache/python_bool/l4/...main.cpp` contains only the `* 2.0`
  branch.

With `TORCH_LOGS=recompiles`, switching from `True` to `False` logs a guard
failure:

```text
Recompiling function forward
triggered by the following guard failure(s):
- is_decode == True
```

For `torch.cond` with a bool tensor predicate, only one wrapper was generated:

- `torch_compile_cache/torch_cond_tensor/if/...main.cpp`

Important lines:

```text
31: aoti_torch_item_bool(arg0_1, &arg0_1_scalar)
32: if (arg0_1_scalar) {
33:     // subgraph: true_graph_0
47:     // subgraph: false_graph_0
```

Trying to write Python control flow over `is_decode.item()` is not the same as
`torch.cond`; with `fullgraph=True` it fails as data-dependent Python control
flow:

```text
Could not guard on data-dependent expression Eq(u0, 1)
Caused by: if is_decode.item():
```

## Practical Recommendation

For prefill/decode, the simplest robust design is usually one of these:

1. Compile two explicit entry points, `forward_prefill` and `forward_decode`,
   and dispatch before calling compiled code. This matches how Dynamo naturally
   specializes Python control flow and usually gives better optimized graphs.
2. Use one `torch.cond` entry point only if you truly need a single compiled
   artifact with runtime branch selection. The predicate must be a tensor, and
   both branches need to satisfy `torch.cond` restrictions, including compatible
   outputs.

