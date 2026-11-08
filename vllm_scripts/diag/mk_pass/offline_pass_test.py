"""Offline harness for the qnorm_qb_uk pattern pass.

Rebuilds the auto_functionalized-form FX graph from an annotated AOT dump
(the exact graph form the inductor custom post-grad pass receives), runs
the pass, then applies the same inductor post-pass housekeeping
(stable_topological_sort + FakeTensorUpdater.incremental_update + lint).

Usage: .venv/bin/python vllm_scripts/diag/mk_pass/offline_pass_test.py <dump.txt> [expected_fused_count]

Dumps come from VLLM_XCPU_DUMP_POST_GRAD_GRAPH (live post-grad graphs,
print_readable form) or the aotautograd cache's aot_forward_graph_str.
"""

import ast
import logging
import operator
import re
import sys

_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO + "/vllm-xcpu-plugin")
sys.path.insert(0, _REPO)

import torch
from torch.fx import Graph, GraphModule
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_xcpu  # noqa: F401  registers torch.ops.torch_xcpu + fake impls
import torch_mcpu  # noqa: F401  registers the mcpu device
import torch_mpi_ext  # noqa: F401  ops referenced by DP/EP graphs

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

DTYPES = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f16": torch.float16,
    "f64": torch.float64,
    "i64": torch.int64,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8": torch.int8,
    "u8": torch.uint8,
    "i1": torch.bool,
    "b8": torch.bool,
    "f8e4m3fn": torch.float8_e4m3fn,
    "f8e5m2": torch.float8_e5m2,
}

ANNO_RE = re.compile(r"^(\w+)(?::\s*\"([^\"]+)\")?\s*=\s*(.*)$")
META_RE = re.compile(r"^([a-z0-9]+)\[([^\[\]]+)\]\[([^\[\]]+)\]([a-z0-9:]+)$")
SYM_RE = re.compile(r"^Sym\((s\d+)\)$")
DEVICE_RE = re.compile(r"^device\(type='([^']+)', index=(-?\d+)\)$")

SYM_VALS: dict[str, int] = {}

symint_ops = {
    "mul": operator.mul,
    "add": operator.add,
    "sub": operator.sub,
    "floordiv": operator.floordiv,
    "truediv": operator.truediv,
    "mod": operator.mod,
}

INFIX_RE = re.compile(r"^(\w+) (//|==|!=|<=|>=|[-+*/%<>]) (-?\w+)$")
INFIX_FN = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "==": operator.eq,
    "!=": operator.ne,
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}


def eval_size(expr: str) -> int:
    try:
        return int(expr)
    except ValueError:
        pass
    try:
        return int(eval(expr, {"__builtins__": {}}, dict(SYM_VALS)))
    except NameError:
        # SymPy-printed dim: Max(0, Min(s72 + ((s72//(-4))), ...))
        import sympy

        parsed = sympy.sympify(
            expr, locals={"Max": sympy.Max, "Min": sympy.Min}
        )
        return int(parsed.subs(SYM_VALS))


def parse_meta(anno: str):
    """'bf16[s72, 2048][2048, 1]mcpu:0' -> (shape, dtype, device) | None."""
    m = META_RE.match(anno)
    if m is None:
        return None
    dtype = DTYPES[m.group(1)]
    shape = tuple(eval_size(x) for x in split_top_commas(m.group(2)))
    device = torch.device(m.group(4))
    return shape, dtype, device


def split_top_commas(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_scalar(text: str, env):
    text = text.strip()
    if text.startswith(("'", '"')):
        return ast.literal_eval(text)
    if text in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[text]
    dm = DEVICE_RE.match(text)
    if dm:
        return torch.device(dm.group(1), int(dm.group(2)))
    if text.startswith("["):
        return [
            unwrap(parse_scalar(x, env)) for x in split_top_commas(text[1:-1])
        ]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("torch."):
        obj = torch
        for part in text.split(".")[1:]:
            obj = getattr(obj, part)
        return obj
    return env[text]  # node name


def resolve_target(path: str):
    if path == "getitem":
        return operator.getitem
    if path in symint_ops:
        return symint_ops[path]
    obj = torch
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def unwrap(v):
    return v.node if isinstance(v, _P) else v


def _try_int_meta(node, args):
    """FakeSymInt nodes need an int meta val (get_fake maps misses to None)."""
    vals = []
    for a in args:
        v = a.meta.get("val") if isinstance(a, torch.fx.Node) else a
        if isinstance(v, torch.Tensor) or not isinstance(v, (int, bool)):
            return
        vals.append(v)
    if node.target in INFIX_FN.values() or node.target is operator.neg:
        node.meta["val"] = node.target(*vals)
    elif node.target is torch.sym_max:
        node.meta["val"] = max(vals)
    elif node.target is torch.sym_min:
        node.meta["val"] = min(vals)


class _P:
    """Name-bound value: either an fx node or a plain constant."""

    __slots__ = ("node", "value")

    def __init__(self, node=None, value=None):
        self.node = node
        self.value = value

    def __getitem__(self, idx):
        if isinstance(self.node, torch.fx.Node):
            return _P(graph.create_node(
                "call_function", operator.getitem, (self.node, idx)
            ))
        return _P(value=self.value[idx])


graph: Graph = None  # set in build()


def cut_suffix(line: str) -> str:
    """Drop the trailing ';  x = y = None' DCE bookkeeping suffix."""
    quote, depth = "", 0
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ";" and depth == 0:
            return line[:i].rstrip()
    return line


def eval_expr(expr: str, env):
    try:
        return _eval_expr(expr, env)
    except Exception as e:
        raise type(e)(f"{e} [expr={expr!r}]") from None


def _eval_expr(expr: str, env):
    expr = expr.strip()
    try:
        return _P(value=int(expr))
    except ValueError:
        pass
    m = re.match(r"^-(\w+)$", expr)
    if m:  # unary minus on a node: -floordiv_5
        assert not m.group(1).isdigit() and m.group(1) in env, (
            f"unary minus on non-node {expr!r}"
        )
        a = unwrap(env[m.group(1)])
        node = graph.create_node("call_function", operator.neg, (a,))
        _try_int_meta(node, [a])
        return _P(node)
    m = re.match(r"^(\w+)((?:\[\d+\])+)$", expr)
    if m:  # subscript form: auto_functionalized_12[1]
        base = env[m.group(1)]
        for idx in re.findall(r"\[(\d+)\]", m.group(2)):
            base = base[int(idx)]
        return base
    m = re.match(r"^getitem\((\w+), (\d+)\)$", expr)
    if m:
        return env[m.group(1)][int(m.group(2))]
    m = INFIX_RE.match(expr)
    if m:  # symint arithmetic: add_9 = arg0_1 + 15 (either side may be const)
        a = unwrap(parse_scalar(m.group(1), env))
        b = unwrap(parse_scalar(m.group(3), env))
        node = graph.create_node(
            "call_function", INFIX_FN[m.group(2)], (a, b)
        )
        _try_int_meta(node, [a, b])
        return _P(node)
    m = re.match(r"^([\w.]+)\((.*)\)$", expr, re.S)
    assert m, f"unparsed expr: {expr!r}"
    target = resolve_target(m.group(1))
    body = m.group(2)
    args, kwargs = [], {}
    if body:
        for part in split_top_commas(body):
            kv = re.match(r"^(\w+) = (.*)$", part, re.S)
            if kv and " = " in part:
                kwargs[kv.group(1)] = unwrap(parse_scalar(kv.group(2), env))
            else:
                args.append(unwrap(parse_scalar(part, env)))
    node = graph.create_node("call_function", target, tuple(args), kwargs)
    _try_int_meta(node, args)
    return _P(node)


def build(path: str) -> tuple[GraphModule, int]:
    global graph
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    lines = open(path).read().splitlines()
    graph = Graph()
    env: dict[str, _P] = {}
    i, n_sig, sym_count = 0, 0, 0
    # signature placeholders
    assert lines[0].startswith("class"), lines[0]
    i = 1
    assert lines[i].strip().startswith("def forward"), lines[i]
    def_line = lines[i].strip()
    sig_re = re.compile(r"^\s*(\w+): \"([^\"]+)\"")
    sig_args = re.findall(r"(\w+): \"([^\"]+)\"", def_line)
    if def_line.rstrip().endswith("):") and sig_args:
        # print_readable form: whole signature on the def line
        sig_entries = sig_args
        i += 1
    else:
        sig_entries = []
        i += 1
        single_line_sig = False
        while i < len(lines):
            line = lines[i]
            if line.rstrip().endswith("):"):
                if sig_re.match(line):
                    pass
                else:
                    i += 1
                    break
            m = sig_re.match(line)
            if m is None:
                if line.strip() in ("self,", "self"):
                    i += 1
                    continue
                raise AssertionError(f"bad signature line {i}: {line!r}")
            sig_entries.append((m.group(1), m.group(2)))
            i += 1
    for name, anno in sig_entries:
        sm = SYM_RE.match(anno)
        if sm:
            sym = sm.group(1)
            if sym not in SYM_VALS:
                SYM_VALS[sym] = 32
                sym_count += 1
            node = graph.create_node("placeholder", name, (), {})
            node.meta["val"] = SYM_VALS[sym]
        else:
            shape, dtype, device = parse_meta(anno)
            with fake_mode:
                t = torch.empty(shape, dtype=dtype, device=device)
            node = graph.create_node("placeholder", name, (), {})
            node.meta["val"] = t
        env[name] = _P(node=node)
        n_sig += 1
    # body
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line.strip() or line.strip().startswith("#"):
            continue
        s = line.strip()
        if s.startswith("return"):
            inner_lines = []
            if s.endswith(")"):
                inner_lines.append(s[s.index("(") + 1 : s.rindex(")")])
            else:
                while True:
                    line2 = lines[i].strip()
                    i += 1
                    if line2 == ")":
                        break
                    inner_lines.append(line2)
            inner = " ".join(inner_lines)
            vals = [
                unwrap(parse_scalar(x, env))
                for x in split_top_commas(inner)
                if x and not x.startswith("#")
            ]
            graph.create_node("output", "output", (tuple(vals),), {})
            break
        m = ANNO_RE.match(s)
        assert m, f"unparsed body line {i}: {s!r}"
        name, anno, expr = m.group(1), m.group(2), cut_suffix(m.group(3))
        val = eval_expr(expr, env)
        env[name] = val
        if anno is not None and isinstance(val, _P) and val.node is not None:
            if anno.startswith("Sym("):
                continue  # symint-valued node; no fake tensor meta needed
            parsed = parse_meta(anno)
            assert parsed is not None, f"bad meta {anno!r}"
            shape, dtype, device = parsed
            with fake_mode:
                t = torch.empty(shape, dtype=dtype, device=device)
            val.node.meta["val"] = t
    gm = GraphModule(torch.nn.Module(), graph)
    gm.delete_all_unused_submodules()
    print(f"placeholders: {n_sig}  syms fixed: {sym_count} -> {SYM_VALS}")
    return gm, fake_mode


def M_args(fused_node):
    for a in fused_node.args + tuple(fused_node.kwargs.values()):
        if isinstance(a, torch.fx.Node):
            yield a


def main():
    path = sys.argv[1]
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else None

    from vllm_xcpu_plugin.compile_passes.qnorm_qb_uk import (
        replace_with_fused_mla_qnorm_qb_uk,
    )
    from torch._inductor.fx_utils import FakeTensorUpdater
    from torch._inductor.fx_passes.post_grad import stable_topological_sort

    gm, fake_mode = build(path)
    g = gm.graph
    n_before = len(list(g.nodes))
    af_before = sum(
        1 for n in g.nodes
        if n.target is torch.ops.higher_order.auto_functionalized
    )
    # Mirror inductor post_grad(): the updater snapshots hashes BEFORE the
    # custom pass runs, so pass-created nodes get re-propagated below.
    if not hasattr(gm, "fake_mode"):
        gm.fake_mode = fake_mode
    updater = FakeTensorUpdater(gm)
    replaced = replace_with_fused_mla_qnorm_qb_uk(g)
    af_after = sum(
        1 for n in g.nodes
        if n.target is torch.ops.higher_order.auto_functionalized
    )
    fused_af = sum(
        1 for n in g.nodes
        if n.target is torch.ops.higher_order.auto_functionalized
        and n.args[0] is torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache.default
    )
    stable_topological_sort(g)
    from torch._inductor.virtualized import V

    with V.set_fake_mode(fake_mode):
        try:
            updated = updater.incremental_update()
        except Exception as e:
            # Replay each node's fake call to name the failing one.
            from torch.fx import Node as _N_NODE

            for n in g.nodes:
                if n.op != "call_function":
                    continue
                try:
                    mapped_args = [
                        a.meta["val"] if isinstance(a, _N_NODE) else a
                        for a in n.args
                    ]
                    mapped_kwargs = {
                        k: v.meta["val"] if isinstance(v, _N_NODE) else v
                        for k, v in n.kwargs.items()
                    }
                    n.target(*mapped_args, **mapped_kwargs)
                except Exception as e2:
                    print(f"  failing node: {n.name} = {n.target}")
                    print(f"    args: {n.args}")
                    print(f"    kwargs: {n.kwargs}")
                    print(f"    error: {e2}")
                    break
            raise
    g.lint()
    n_after = len(list(g.nodes))
    # Every new node must carry a fake val with the expected shapes.
    fused_nodes = [
        n
        for n in g.nodes
        if n.target is torch.ops.higher_order.auto_functionalized
        and n.args[0] is torch.ops.torch_xcpu.fused_mla_qnorm_qb_uk_rope_cache.default
    ]
    for f in fused_nodes:
        for arg in M_args(f):
            assert "val" in arg.meta, f"{arg.name} has no fake val"
    if fused_nodes:
        f0 = fused_nodes[0]
        for kw in ("qkv", "slots", "out_q"):
            v = f0.kwargs[kw].meta["val"]
            print(f"  fused arg {kw}: {tuple(v.shape)} {v.dtype}")
        for kw in ("q_c_norm", "kv_cache", "out_q"):
            gi = f0.kwargs[kw]
        q_c = next(
            u for u in fused_nodes[0].users
            if u.target is operator.getitem and u.args[1] == 3
        )
        print(f"  fused getitem q_c_norm: {tuple(q_c.meta['val'].shape)}")
    print(f"nodes: {n_before} -> {n_after}   af nodes: {af_before} -> {af_after}")
    print(f"pass replaced: {replaced}   fused ops in graph: {fused_af}")
    print(f"FakeTensorUpdater updated: {updated} nodes")
    if expected is not None:
        assert replaced == expected, f"expected {expected}, got {replaced}"
        print(f"OK: matched expected count {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
