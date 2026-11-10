# Kimi-K3-INT4 DLC 适配

当前目录保存 Kimi-K3-INT4 在 DLC + vLLM 0.30.1 基线上的适配准备材料。

- `适配基线与差距分析_2026-10-28.md`：环境、模型结构、GLM 复用结论、硬阻塞和首批工作项。
- `切片模型_TP8_2026-10-28.md`：单机 8 卡 TP8 的 24 层真实权重切片、生成与验证说明。
- `算子移植与原生回退_2026-10-28.md`：K3 算子到 DLC kernel/native fallback 的映射、验证和剩余差距。
- `op_requests/`：按 `DLC_Custom_Kernel/.github/ISSUE_TEMPLATE/operator_request.yml` 编写的算子需求草稿。
- `/root/workspace/work-tests/kimi_k3/probe_kda_shapes.py`：KDA 几何探测脚本。
- `/root/workspace/work-tests/kimi_k3/README.md`：KDA 探针运行方法和安全约束。
- `/root/workspace/work-tests/kimi_k3/audit_checkpoint.py`：只读 safetensors 头的分片与量化完整性审计。
- `/root/workspace/work-tests/kimi_k3/reports/checkpoint_audit_20261028.json`：本次审计结果。

vLLM 与 vllm-cl 的适配分支统一为 `release-kimi-k3-int4`。
