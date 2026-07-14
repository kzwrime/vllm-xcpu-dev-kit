from __future__ import annotations

import difflib
import html
import json
import re
from collections import Counter
from pathlib import Path

from .model import Comparison, Inventory, Kernel, Match, jsonable


STATUS_LABELS = {
    "unchanged": "未变化",
    "modified": "原位修改",
    "moved": "移动并修改",
    "renamed": "重命名并修改",
    "moved_renamed": "移动、重命名并修改",
    "moved_source_exact": "移动（源码相同）",
    "renamed_source_exact": "重命名（源码相同）",
    "moved_renamed_source_exact": "移动并重命名（源码相同）",
    "modified_semantic_exact": "仅格式/注释变化",
    "moved_semantic_exact": "移动（语义相同）",
    "renamed_semantic_exact": "重命名（语义相同）",
    "moved_renamed_semantic_exact": "移动并重命名（语义相同）",
    "added": "新增",
    "removed": "删除",
}


def _label(status: str) -> str:
    return STATUS_LABELS.get(status, status.replace("_", " "))


def _location(kernel: Kernel | None) -> str:
    return "—" if kernel is None else f"{kernel.path}:{kernel.definition_line} · {kernel.qualname}"


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return value[:100] or "kernel"


def _unified(match: Match) -> str:
    old_source = match.old.source.splitlines(keepends=True) if match.old else []
    new_source = match.new.source.splitlines(keepends=True) if match.new else []
    old_name = _location(match.old)
    new_name = _location(match.new)
    lines = difflib.unified_diff(old_source, new_source, fromfile=old_name, tofile=new_name, n=3)
    return "".join(lines)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _inventory_markdown(inventory: Inventory, title: str) -> str:
    repo = inventory.repository
    rows = [
        f"# {title}", "",
        f"- 仓库：`{repo.path}`",
        f"- Git：`{repo.revision}` (`{repo.commit[:12]}`, `{repo.branch}`)，工作树{'有修改' if repo.dirty else '干净'}",
        f"- 扫描 Python 文件：{inventory.scanned_python_files}",
        f"- Triton JIT kernel：{len(inventory.kernels)}", "",
        "| # | 文件与行号 | 限定函数名 | 行数 | 装饰器 |", "|---:|---|---|---:|---|",
    ]
    for index, kernel in enumerate(inventory.kernels, 1):
        decorators = "<br>".join(
            f"`{item.replace('|', '&#124;').replace(chr(10), '<br>')}`"
            for item in kernel.decorators
        )
        rows.append(
            f"| {index} | `{kernel.path}:{kernel.definition_line}` | `{kernel.qualname}` | "
            f"{kernel.line_count} | {decorators} |"
        )
    if inventory.issues:
        rows.extend(["", "## 扫描问题", "", "| 文件 | 行 | 问题 |", "|---|---:|---|"])
        for issue in inventory.issues:
            rows.append(f"| `{issue.path}` | {issue.line or '—'} | {issue.message} |")
    return "\n".join(rows) + "\n"


def write_inventory(inventory: Inventory, output: str | Path, title: str = "Triton kernel 清单") -> None:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "inventory.json", inventory)
    (target / "inventory.md").write_text(_inventory_markdown(inventory, title), encoding="utf-8")


def _summary(comparison: Comparison) -> Counter[str]:
    result: Counter[str] = Counter()
    for match in comparison.matches:
        result[match.status] += 1
        result["changed" if match.changed else "unchanged_total"] += 1
        if match.method in {"semantic_exact", "source_exact", "same_name", "fuzzy"} and (
            match.old and match.new and (match.old.path != match.new.path or match.old.name != match.new.name)
        ):
            result["non_direct"] += 1
        if match.confidence == "low" and match.old and match.new:
            result["low_confidence"] += 1
    return result


def _prepare_diffs(comparison: Comparison, target: Path) -> None:
    diff_dir = target / "diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    # Reusing an output directory must not leave stale diffs from a previous
    # run whose matching decisions or kernel count were different.
    for stale in diff_dir.glob("*.diff"):
        stale.unlink()
    number = 0
    for match in comparison.matches:
        if not match.changed:
            continue
        number += 1
        kernel = match.new or match.old
        filename = f"{number:04d}_{_slug(kernel.name)}.diff"
        content = _unified(match)
        if not content:
            content = (
                f"# No textual hunk. Location/name metadata changed.\n"
                f"# old: {_location(match.old)}\n# new: {_location(match.new)}\n"
            )
        (diff_dir / filename).write_text(content, encoding="utf-8")
        match.diff_file = f"diffs/{filename}"


def _markdown(comparison: Comparison) -> str:
    summary = _summary(comparison)
    old, new = comparison.old.repository, comparison.new.repository
    lines = [
        "# vLLM Triton kernel 差异报告", "",
        f"> 旧仓库：`{old.path}` · `{old.revision}` · `{old.commit[:12]}`  ",
        f"> 新仓库：`{new.path}` · `{new.revision}` · `{new.commit[:12]}`", "",
        "## 总览", "",
        "| 指标 | 数量 |", "|---|---:|",
        f"| 旧版本 kernel | {len(comparison.old.kernels)} |",
        f"| 新版本 kernel | {len(comparison.new.kernels)} |",
        f"| 产生变化 | {summary['changed']} |",
        f"| 未变化 | {summary['unchanged_total']} |",
        f"| 非直接匹配（移动/重命名） | {summary['non_direct']} |",
        f"| 低置信度匹配（需要人工确认） | {summary['low_confidence']} |", "",
        "差异行统计中，`+/-` 是新增/删除行数；`Δ` 对 replace 块取两侧较大值，表示受影响行数，避免一次修改被重复计算。", "",
        "## 变化分类", "",
        "| 分类 | 数量 |", "|---|---:|",
    ]
    for status, count in sorted(
        ((status, count) for status, count in summary.items()
         if status not in {"changed", "unchanged_total", "non_direct", "low_confidence", "unchanged"}),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| {_label(status)} | {count} |")
    lines.extend([
        "",
        "## 所有产生变化的 kernel", "",
        "| 状态 | 旧位置 | 新位置 | 差异行 | 匹配依据 | 置信度 | 查看 |",
        "|---|---|---|---:|---|---|---|",
    ])
    for match in comparison.matches:
        if not match.changed:
            continue
        stats = f"+{match.stats.added}/-{match.stats.deleted} (Δ{match.stats.changed})"
        link = f"[diff]({match.diff_file})" if match.diff_file else "—"
        lines.append(
            f"| {_label(match.status)} | `{_location(match.old)}` | `{_location(match.new)}` | "
            f"{stats} | `{match.method}` ({match.score:.3f}) | {match.confidence} | {link} |"
        )
    uncertain = [m for m in comparison.matches if m.confidence == "low" and m.old and m.new]
    if uncertain:
        lines.extend(["", "## 需要人工确认的匹配", "",
                      "这些项目仍按最佳候选生成 diff，但不应被当作确定的移动/重命名。", "",
                      "| 旧位置 | 最佳新位置 | 得分 | 其它候选位置 |", "|---|---|---:|---|"])
        for match in uncertain:
            alternatives = ", ".join(f"`{item}` ({score:.3f})" for item, score in match.alternatives) or "—"
            lines.append(f"| `{_location(match.old)}` | `{_location(match.new)}` | {match.score:.3f} | {alternatives} |")
    lines.extend(["", "## 完整清单", "",
                  "- [旧版本 kernel 清单](inventory_old.md)",
                  "- [新版本 kernel 清单](inventory_new.md)",
                  "- [机器可读比较结果](comparison.json)",
                  "- [交互式 HTML 报告](report.html)", ""])
    return "\n".join(lines)


def _html_report(comparison: Comparison) -> str:
    summary = _summary(comparison)
    rows: list[str] = []
    details: list[str] = []
    for index, match in enumerate((m for m in comparison.matches if m.changed), 1):
        anchor = f"kernel-{index}"
        search = " ".join(filter(None, [match.status, match.method,
                                         match.old.name if match.old else "",
                                         match.new.name if match.new else "",
                                         match.old.path if match.old else "",
                                         match.new.path if match.new else ""]))
        rows.append(
            f'<tr data-search="{html.escape(search.lower())}"><td>{html.escape(_label(match.status))}</td>'
            f'<td><code>{html.escape(_location(match.old))}</code></td>'
            f'<td><code>{html.escape(_location(match.new))}</code></td>'
            f'<td>+{match.stats.added}/-{match.stats.deleted} (Δ{match.stats.changed})</td>'
            f'<td>{html.escape(match.method)} · {match.score:.3f} · {match.confidence}</td>'
            f'<td><a href="#{anchor}">展开</a> · <a href="{html.escape(match.diff_file or "")}">diff</a></td></tr>'
        )
        details.append(
            f'<details id="{anchor}"><summary>{index}. {html.escape(_label(match.status))} — '
            f'{html.escape((match.new or match.old).name)}</summary>'
            f'<p><b>旧：</b><code>{html.escape(_location(match.old))}</code><br>'
            f'<b>新：</b><code>{html.escape(_location(match.new))}</code><br>'
            f'<b>匹配：</b>{html.escape(match.method)} / {match.score:.3f} / {match.confidence}</p>'
            f'<pre>{html.escape(_unified(match) or "（文本相同，仅路径或名称发生变化）")}</pre></details>'
        )
    old, new = comparison.old.repository, comparison.new.repository
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>vLLM Triton kernel 差异报告</title><style>
body{{font:14px/1.55 system-ui,sans-serif;margin:2rem auto;max-width:1600px;padding:0 1rem;color:#202124}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d8dee4;padding:.45rem;text-align:left;vertical-align:top}} th{{background:#f6f8fa;position:sticky;top:0}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap}} .card{{border:1px solid #d8dee4;border-radius:8px;padding:.7rem 1rem;min-width:130px}}
.number{{font-size:1.5rem;font-weight:700}} input{{padding:.6rem;width:min(600px,90%);margin:1rem 0}}
details{{margin:1rem 0;border:1px solid #d8dee4;border-radius:6px;padding:.6rem}} summary{{cursor:pointer;font-weight:600}}
pre{{overflow:auto;background:#0d1117;color:#e6edf3;padding:1rem;border-radius:6px;max-height:700px}} a{{color:#0969da}}
</style></head><body>
<h1>vLLM Triton kernel 差异报告</h1>
<p><b>旧：</b><code>{html.escape(old.path)}</code> · {html.escape(old.revision)}<br>
<b>新：</b><code>{html.escape(new.path)}</code> · {html.escape(new.revision)}</p>
<div class="cards"><div class="card">旧 kernel<div class="number">{len(comparison.old.kernels)}</div></div>
<div class="card">新 kernel<div class="number">{len(comparison.new.kernels)}</div></div>
<div class="card">产生变化<div class="number">{summary['changed']}</div></div>
<div class="card">未变化<div class="number">{summary['unchanged_total']}</div></div>
<div class="card">非直接匹配<div class="number">{summary['non_direct']}</div></div>
<div class="card">低置信度<div class="number">{summary['low_confidence']}</div></div></div>
<input id="filter" placeholder="搜索路径、函数名、状态或匹配方法…" oninput="filterRows(this.value)">
<table><thead><tr><th>状态</th><th>旧位置</th><th>新位置</th><th>差异行</th><th>匹配</th><th>查看</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>逐 kernel 差异</h2>{''.join(details)}
<script>function filterRows(q){{q=q.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.dataset.search.includes(q));}}</script>
</body></html>"""


def write_comparison(comparison: Comparison, output: str | Path) -> None:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    _prepare_diffs(comparison, target)
    _write_json(target / "inventory_old.json", comparison.old)
    _write_json(target / "inventory_new.json", comparison.new)
    _write_json(target / "comparison.json", comparison)
    (target / "inventory_old.md").write_text(_inventory_markdown(comparison.old, "旧版本 Triton kernel 清单"), encoding="utf-8")
    (target / "inventory_new.md").write_text(_inventory_markdown(comparison.new, "新版本 Triton kernel 清单"), encoding="utf-8")
    (target / "report.md").write_text(_markdown(comparison), encoding="utf-8")
    (target / "report.html").write_text(_html_report(comparison), encoding="utf-8")
