# vLLM DLC 上游对齐策略

## 1. 仓库关系

vLLM DLC 是独立的 DLC TPU 插件仓库，不通过 merge 或 rebase 合入 vLLM 的 Git 历史。
它针对一个确定的上游 vLLM 版本和 commit 完成接口适配，并通过
`vllm_cl/vllm_alignment.yaml` 保存该组合。

上游仓库：

```text
git@github.com:ChipLTech/vllm.git
```

## 2. 当前对齐基线

当前仓库记录的对齐信息如下：

| 项目 | 当前值 |
| --- | --- |
| 上游 vLLM 版本 | `v0.23.1rc0` |
| 上游 vLLM commit | `71e2bbbc39f36481d9896eaf92efce00a95ad27a` |
| 完成对齐时的 vLLM DLC 分支 | `v0.23.1rc0` |
| 对齐记录日期 | `2026-07-08` |

完整 commit SHA 是可复现对齐关系的依据。上游分支可以继续移动，因此不能只记录分支名。

## 3. 机器可读记录

当前 `vllm_cl/vllm_alignment.yaml` 的结构为：

```yaml
vllm:
  version: v0.23.1rc0
  commit: 71e2bbbc39f36481d9896eaf92efce00a95ad27a
vllm_cl:
  branch: v0.23.1rc0
  updated_at: 2026-07-08
```

字段含义：

- `vllm.version`：本次适配所属的上游版本线；
- `vllm.commit`：实际完成适配和验证的上游完整 SHA；
- `vllm_cl.branch`：完成该次对齐时使用的 vLLM DLC 分支；
- `updated_at`：完成对齐并更新记录的日期。

## 4. 后续升级方式

每次将 vLLM DLC 对齐到新的上游版本或 commit 时，应在同一个升级 PR 中完成：

1. 确定旧、新 vLLM 完整 commit，以及目标所属版本线；
2. 比较两个 commit 之间的接口和行为变化；
3. 更新受影响的 vLLM DLC platform、worker、model runner、patch 和算子适配；
4. 必要时更新 `vllm_cl/patch_import_manifest.yaml`；
5. 完成目标场景验证；
6. 将新的 version、commit、vLLM DLC 分支和日期写入 `vllm_alignment.yaml`；
7. 将代码适配、测试结果和 alignment 更新一起合入。

在新组合验证通过之前，不应提前更新 alignment。`main` 可以继续开发，但每一个声称可用的组合
都必须能落到一个确定的 vLLM commit 和一个确定的 vLLM DLC commit/tag。

## 5. 与发布版本的关系

`vllm_alignment.yaml` 固定发布时使用的上游 vLLM commit；vLLM DLC 自身的 Git tag 固定插件
代码。两者组合后才能唯一复现一个 Release：

```text
vLLM DLC tag
    + vllm_alignment.yaml 中的上游 vLLM commit
    = 可复现的 vLLM DLC 发布组合
```
