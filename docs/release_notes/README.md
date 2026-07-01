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

该命令还会同时生成 `.release/repository_versions_currently.json`，其中 `repositories[].version` 是各仓库当前 `HEAD`。如需更改该文件位置，可使用 `--current-versions-output`。

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
- `vllm` 除源码包外，还会生成 patch。patch 范围从清单中记录的上一次 `version` 之后的第一个 commit 到当前 `HEAD`。
- 脚本会生成 `.release/publish/repository_versions_currently.json`，记录本次发布时各仓库的当前 `HEAD`。
- `vllm-xcpu-dev-kit` 的清单路径是 `./`，解析后等于主仓库根目录，也会通过 clone 后打包的方式生成 `vllm-xcpu-dev-kit_*.tar.gz`。源码包来自主仓库已提交的 `HEAD`，不会包含工作区里的未跟踪文件或 `.release/publish` 中的本地发布产物。

推荐顺序：

1. 使用 `--generate-template` 生成更新说明和当前版本快照。
2. 检查并编辑 `docs/release_notes/YYYYMMDD 更新说明.md`。
3. 在 `.release/repository_versions.json` 仍表示上一次发版基线时运行 `scripts/package_release_publish.py`，否则 `vllm` patch 可能因为没有新增 commit 而失败。
4. 确认发布产物后运行 `--update-versions` 推进 `.release/repository_versions.json`。
