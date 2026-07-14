# vLLM Triton Kernel Analyzer

一个不导入 vLLM 或 Triton 的静态分析器，用于清点 Git 工作树中的
`@triton.jit` Python kernel，并比较两个快速演进、可能发生目录重构的 vLLM
源码仓库。

## 能得到什么

- 完整 kernel 清单：仓库 Git 版本、文件路径、限定函数名、起止行、签名、装饰器、源码和指纹。
- 多阶段匹配：同路径/同限定名、源码或 AST 语义指纹、同函数名、源码与路径模糊匹配。
- 明确报告移动、重命名和低置信度/歧义匹配，避免把猜测伪装成确定结论。
- 每个变化 kernel 的 unified diff 与 `+新增/-删除/Δ受影响` 行数。
- 便于浏览的 Markdown、可搜索并内嵌 diff 的 HTML，以及完整 JSON 数据。

扫描默认覆盖仓库当前工作树中的所有 `.py` 文件，包括已跟踪、未跟踪文件以及
`vllm/`、`tests/`、`benchmarks/`。它跳过 `.git`、虚拟环境、构建和缓存目录。
因此报告准确描述的是当前工作树，不一定等同于 `HEAD`；报告会记录 dirty 状态。

## 使用

在套件根目录运行：

```bash
python -m tools.vllm_triton_kernel_analyzer scan \
  /path/to/vllm -o tools/vllm_triton_kernel_analyzer/reports/one-version

python -m tools.vllm_triton_kernel_analyzer compare \
  /path/to/old/vllm /path/to/new/vllm \
  -o tools/vllm_triton_kernel_analyzer/reports/old-to-new
```

例如本套件的两份源码：

```bash
python -m tools.vllm_triton_kernel_analyzer compare \
  /shared/vllm-xcpu-dev-kit-0.19/vllm \
  /shared/vllm-xcpu-dev-kit-0.24/vllm \
  -o tools/vllm_triton_kernel_analyzer/reports/v0.19-to-v0.24
```

可重复使用 `--exclude DIR_NAME` 排除额外目录。`--fuzzy-threshold` 可调节移动且
重命名的候选阈值，默认 `0.58`；提高它会减少误匹配，也会产生更多“新增/删除”。

## 产物布局

```text
report.html             可搜索、逐项展开 diff 的主报告
report.md               适合代码审查和文本检索的主报告
inventory_old.md        旧版本完整清单
inventory_new.md        新版本完整清单
comparison.json         完整机器可读比较数据
inventory_old.json      带源码和指纹的旧版本数据
inventory_new.json      带源码和指纹的新版本数据
diffs/*.diff            每个变化 kernel 的独立 unified diff
```

## 识别与匹配边界

扫描器使用 Python AST，并通过 `import triton`、`import triton as ...`、
`from triton import jit as ...` 或 vLLM 自己的
`from vllm.triton_utils import triton as ...` 转发导入确认装饰器来源，所以不会把 Numba 的 `@jit`
当作 Triton kernel。无法被当前 Python 解释器解析的文件会进入“扫描问题”，不会
静默忽略。

匹配依次消耗候选，保证一个旧 kernel 最多对应一个新 kernel：

1. 相同相对路径和限定函数名；
2. 完全相同源码或忽略函数名、位置、注释和排版后的 AST 语义指纹；
3. 相同函数名下按函数体 token 顺序和路径相似度选择；
4. 对剩余项做源码、函数名和路径的综合模糊匹配。

模糊匹配本质上是启发式结果。报告中的 `confidence=low` 和相近候选必须人工确认；
如果希望保守审阅，可提高阈值并把未匹配的删除/新增项成对检查。

## 测试

```bash
python -m unittest discover -s tools/vllm_triton_kernel_analyzer/tests -v
```
