# Dev Log 规范

## 文件命名格式

e.g. 20260918_Qwen3.8-27B-Dflash2-aten-ops-opt.md

## 元数据

e.g.

```
---
date: 2026-09-20
type: feature
status: completed
scope: auth
author: agent
related:
  project1:
    branch: xxx
    commit: "abc1234"
    commit: "abc5678"
    pr: ...
  project2:
    branch: xxx
    commit: "def1234"
    commit: "def5678"
    pr ...
---
```

- commit 不是仓库 head，而是涉及的提交。
- 涉及的 issue / pr 为可选项。

## 章节要求

```
## 1. 背景及分析
## 2. 方案及效果评估
## 3. 影响及约束
## 4. 后续开发
## 5. 附录
```

附录内可以自行补充其他章节。后续开发中需要额外讨论真实 NPU Infra 团队应该做什么。
