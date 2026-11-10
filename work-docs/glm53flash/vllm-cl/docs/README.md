# vLLM DLC 上游对齐与发布

本目录只维护 vLLM DLC 与上游 vLLM 的对齐信息，以及 vLLM DLC 的 Release 发布方式。

- [上游对齐策略](versioning_policy.md)：当前兼容基线、精确 commit 和后续升级如何更新记录。
- [Release 发布流程](release_process.md)：Release 分支、RC/Final/Post tag、构建和制品发布。

当前对齐关系的机器可读来源是 `vllm_cl/vllm_alignment.yaml`。文档用于解释规则；发生冲突时，
必须先核实实际完成验证的 commit，再更新 alignment 和文档，不能仅依赖会移动的分支名。
