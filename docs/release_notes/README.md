# 多仓库版本更新工作流

本目录用于存放版本更新日志。每次发布生成一个 `YYYYMMDD 更新说明.md`，说明正文可以人工编辑，不作为自动化基线来源。

本仓库使用 `.release/repository_versions.json` 作为多仓库版本清单。它的作用类似 Git submodule 在主仓库中记录子仓库 commit：每个子仓库都有一个明确的 `version`，自动化脚本和 CI 均以该文件作为上一次发版状态的唯一来源。

清单维护规则：

- `path` 使用相对主仓库根目录的路径，并显式带 `./` 前缀。
- `version` 使用完整 40 位 commit id。
- 短 hash 由工具运行时计算，不写入清单，减少人工维护负担。

## 生成更新说明模板

```bash
python3 scripts/generate_update_notes.py --generate-template
```

默认 `--release-version` 为当天的 `YYYYMMDD`，并输出到 `docs/release_notes/<release-version> 更新说明.md`。生成文件后，人工补充 `更新重点` 和 `备注`。

每条 commit 末尾会附带该提交修改的文件数、增删行数；二进制文件无法统计行数时会单独标明。

对于 `vllm`，脚本会检查清单中的上一次版本是否为当前 `HEAD` 的祖先：

- 历史连续时，仍按 `<previous-version>..HEAD` 记录提交。
- 跨大版本、从新主线建立开发分支而导致历史不连续时，自动改按 `origin/main..HEAD` 记录当前分支相对 main 的提交，并在说明中标注基线切换。若仓库没有 `origin/main`，则尝试本地 `main`。
- 如需指定其他主线引用，使用 `--vllm-main-ref <ref>`，例如 `--vllm-main-ref upstream/main`。

该命令还会同时生成 `.release/repository_versions_currently.json`，其中 `repositories[].version` 是各仓库当前 `HEAD`。如需更改该文件位置，可使用 `--current-versions-output`。

## `torch_xcpu` 算子变更编写规则

发布说明中的 `torch_xcpu 算子变更` 面向算子开发团队，记录本次约定区间内需要跟进、校验或重新优化的算子变化。它不是 `torch_xcpu` commit 清单的重复摘要，尤其在大版本升级时，不应把新版本正常交付的全部功能都列为交付后变更。

### 统计范围与收录边界

- 先明确统计起点和终点，正文写明日期及终点完整 commit，例如“2026-07-22（含）至 `<40 位 HEAD>`”。若交付后的修改区间与版本清单基线不同，以人工确认的交付区间为准。
- 逐个检查区间内触及 `torch_xcpu_impl/include`、`torch_xcpu_impl/src` 和 `torch_xcpu_impl/backends` 的 commit。重点是从核实现、参数结构、调度和数据布局变化；同时检查 `torch_xcpu/csrc`、`torch_xcpu/ops_defs` 与测试，确认 Torch schema、Python 包装和可验证行为。
- 收录会影响算子实现或接入的变化，包括：新增算子、参数或 schema 变化、dtype/shape/layout/stride 变化、边界条件修复、任务切分或 kernel launch 变化，以及需要算子团队重新校验的性能修改。
- 不要仅根据 commit 标题筛选。一个大型重构提交可能顺带修改其他算子，例如给空输入增加提前返回；必须检查实际 diff，避免遗漏。
- 默认不额外描述正常交付阶段完成的整块功能或重构，例如约定排除的 MoE、FP8、MXFP4 部分；但该模块中若有明确属于交付后区间的独立算子兼容性修改，仍应单独记录。项目负责人明确要求忽略的算子也不写入本节。
- 若一个算子在区间内先新增、后改名或删除，以发布终点的最终状态为准，不把已经删除的中间接口写成可交付接口。

### 算子名称与分类

- 已有算子的功能扩展或 out/functional 变体，按 `torch_xcpu_impl` 中的初始基础算子名归类。例如新增 `fused_add_rms_norm_out`，条目名称仍写 `fused_add_rms_norm`，正文说明新增的接口。
- 具有独立 Torch schema、参数结构以及 `torch_xcpu_impl` 实现文件的新增算子，保留完整名称并标记“新增算子”。例如 `reshape_and_cache_grouped` 应独立成项，不能并入 `reshape_and_cache`。
- 同一基础算子的多个实现文件可以合并说明；不同算子不要因为服务于同一模型或功能而合并成一个模块级条目。例如 `reshape_and_cache`、`reshape_and_cache_grouped`、`reshape_and_cache_mla` 和 `unified_attention` 应分别判断和描述。
- 条目名称使用代码中的实际名称，不使用“DFlash 辅助算子”“GDN 系列优化”等功能模块名称替代算子名。

### 描述内容与证据

每个条目应先写影响等级和最终行为，再列提交与测试。不能只复制 commit 标题。

```markdown
- `operator_name`（新增算子/小修改/中等修改/大修改）：说明最终接口或实现发生了什么、调用方或从核为什么需要跟进
	- commit `<40 位 commit>`：<commit subject>
	- 测试：`torch_xcpu/test/.../test_operator.py`
```

- “小修改”通常是局部边界处理或不改变正常输入契约的修复；“中等修改”通常涉及 dtype、可选参数、stride、shape 支持或调度方式；“大修改”通常涉及 ABI/schema、核心布局或算法路径变化。
- schema 变化要区分高层 Python 包装与底层 `torch.ops` ABI。如果包装层提供默认值或兼容别名，但底层直调仍需修改，必须明确写出。
- layout、dtype、shape 和 stride 变化尽量给出具体契约，例如 tensor shape、参数类型或触发条件；不要只写“兼容新版本”或“支持某模型”。
- commit 使用完整 40 位 hash。一个算子由多个提交共同形成最终行为时，按时间或逻辑顺序全部列出。
- 优先引用直接覆盖该行为的测试文件。若没有专项测试，应明确写“未新增对应的专项测试”，不能用无关测试暗示已经覆盖。
- 段首统一保留测试覆盖声明：测试用于佐证行为，但不代表覆盖全部 shape、dtype、后端和并行组合。

### 完成前检查

1. 枚举统计区间内所有修改过的 `torch_xcpu_impl` 文件，并按初始算子名建立候选清单。
2. 单独枚举新增的 schema、参数结构和实现文件，确认新增算子没有被误并入基础算子。
3. 检查大型提交中的非主题文件，确认没有遗漏顺带发生的算子修复。
4. 对照发布终点代码，删除只存在于中间提交、最终已经撤销或替换的接口描述。
5. 核对每个 commit hash 和测试路径真实存在，并运行 Markdown 格式检查。

## 更新版本清单

确认更新说明和各仓库 HEAD 都是本次要发布的版本后，推进版本清单：

```bash
python3 scripts/generate_update_notes.py --update-versions
```

默认 `--release-date` 为当天的 `YYYY-MM-DD`。

## 一次完成模板生成和版本推进

```bash
python3 scripts/generate_update_notes.py \
  --generate-template \
  --update-versions
```

提交内容通常包括：

- `.release/repository_versions.json`
- `docs/release_notes/YYYYMMDD 更新说明.md`
- 其他发版必要改动

## 打包发布产物

`scripts/package_release_publish.py` 根据 `.release/repository_versions.json`
中的仓库列表生成发布产物，默认输出到 `.release/publish`：

```bash
python3 scripts/package_release_publish.py
```

默认 `--release-version` 为当天的 `YYYYMMDD`，也可以显式指定：

```bash
python3 scripts/package_release_publish.py --release-version 20260626
```

打包行为：

- 对清单中的每个仓库，脚本都会从当前分支 clone 一份源码树，写入 `.release/repository_version.json`，再生成 `<name>_<release-version>_<short-head>.tar.gz`。
- `vllm` 历史连续时，除源码包外还会生成 patch。patch 范围从清单中记录的上一次 `version` 之后的第一个 commit 到当前 `HEAD`。
- `vllm` 跨大版本且清单中的上一次 `version` 不是当前 `HEAD` 的祖先时，不生成 patch；源码包和当前版本清单仍正常生成。
- 脚本会生成 `.release/publish/repository_versions_currently.json`，记录本次发布时各仓库的当前 `HEAD`。
- `vllm-xcpu-dev-kit` 的清单路径是 `./`，解析后等于主仓库根目录，也会通过 clone 后打包的方式生成 `vllm-xcpu-dev-kit_*.tar.gz`。源码包来自主仓库已提交的 `HEAD`，不会包含工作区里的未跟踪文件或 `.release/publish` 中的本地发布产物。

推荐顺序：

1. 使用 `--generate-template` 生成更新说明和当前版本快照。
2. 检查并编辑 `docs/release_notes/YYYYMMDD 更新说明.md`。
3. 在 `.release/repository_versions.json` 仍表示上一次发版基线时运行 `scripts/package_release_publish.py`；若 vLLM 历史连续则生成增量 patch，若跨主线不连续则明确跳过 patch。
4. 确认发布产物后运行 `--update-versions` 推进 `.release/repository_versions.json`。
