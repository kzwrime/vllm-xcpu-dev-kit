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
