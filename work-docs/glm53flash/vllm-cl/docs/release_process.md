# vLLM DLC Release 发布流程

## 1. 当前方式

当前仓库没有自动发布工作流。Python 包版本由 `setuptools_scm` 从 vLLM DLC 的 Git tag
生成，维护者需要手工创建 tag、构建并上传制品。

当前只有 `v0.23.1` 一条版本线，第一次 RC 可以直接从验证完成的 `main` 发布，不必提前
创建 Release 分支。

## 2. 发布第一个 RC

确认 `vllm_cl/vllm_alignment.yaml` 已记录实际验证的上游 vLLM 版本和完整 commit，然后
从干净的 `main` 创建本地 tag：

```bash
git switch main
git pull --ff-only origin main
git status --short
git tag -a v0.23.1rc0 -m "vLLM DLC v0.23.1rc0"
```

构建并检查制品：

```bash
python -m build --no-isolation
python -m twine check dist/*
```

确认包版本为 `0.23.1rc0`，并完成必要验证后再推送 tag：

```bash
git push origin v0.23.1rc0
```

如果构建或验证失败，不推送 tag；修复后重新创建本地 tag。

## 3. 什么时候创建 Release 分支

只有当 `main` 要继续对齐新的上游 vLLM，而 `v0.23.1` 仍需要维护时，才创建 Release
分支。可以直接从已经发布的 RC tag 创建：

```bash
git branch releases/v0.23.1 v0.23.1rc0
git push -u origin releases/v0.23.1
```

之后：

- 新功能和新的上游对齐继续进入 `main`；
- `v0.23.1` 必需的修复 backport 到 `releases/v0.23.1`；
- 新的 RC、正式版和 Post 从该 Release 分支打 tag。

## 4. 后续版本

| 场景 | 新 tag |
| --- | --- |
| RC0 修复后继续验证 | `v0.23.1rc1` |
| RC 验证完成，发布正式版 | `v0.23.1` |
| 正式版后交付低风险修复 | `v0.23.1.post1` |

RC 阶段发现 bug 时发布下一个 RC；只有正式版发布后的低风险修复才使用 Post。已经发布的
tag 不得移动、删除或覆盖。

## 5. 发布制品

将同一 tag 生成的源码包、wheel 和可选镜像上传到团队约定的位置。使用 Python 制品仓库时：

```bash
python -m twine upload --repository-url <repository-url> dist/*
```

发布记录至少保留：

- vLLM DLC tag 和完整 commit；
- `vllm_alignment.yaml` 中的上游 vLLM 版本和完整 commit；
- 制品地址和校验和；
- 构建环境、发布日期和已知限制。

正式制品必须来自带有完整 tag 信息的干净 checkout。版本显示为 `0.0.0` 或非预期
`.devN` 时不能发布。
