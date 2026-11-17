# FP8 KV cache 预研归档

本目录和对应的 `work-tests/fp8_kvcache` 是本机工作资料，已通过工作区 `.git/info/exclude` 排除，不提交到组件仓库。归档时间及原文件 SHA256 见 archive_manifest.json；原证据 JSON 和 PDF 保留采集时的内容与路径。

- [算子需求目录](operator_requests/README.md)：6份可独立发布的issue正文、布局/数值说明及旧G128报告。
- [宣讲HTML](vllm-cl/presentation.html)、[PDF](vllm-cl/presentation.pdf)：此前评审与宣讲归档。
- [原始KV cache设计](vllm-cl/design.md)、[原始数值与布局契约](vllm-cl/kernel_numerics_and_layout.md)：早期固定per-layer scale、D128/P256方案，属于历史预研，不能当作最新group128/group32算子需求。
- [GLM-5.3 Sparse MLA FP8 实施契约](vllm-cl/glm53_sparse_mla_fp8_plan_20261027.md)：按当前 `main` 的 group128/FP32-scale ABI、GLM `D=512/NoPE/P256` 与 DLC 一字节布局更新，是后续接入的主设计入口。
- [vLLM v0.30.1 FP8 KV cache 适配方案](vllm-cl/v0.30.1_adaptation_plan_20261117.md)：对照旧 `fp8-kvcache-main` 原型与 v0.30.1 的 `KVQuantMode`/`customize_spec`/allocator 变化，给出 GQA FP8 的落地步骤与边界。
- [FP8 KV Cache 使用参考](fp8_kv_cache_usage_reference.md)：分离 FP8 values + flat FP32 scales 的 API、容量公式、真机实测、反例和当前限制。
- [Qwen3 低利用率容量与精度验收](qwen3_fp8_kvcache_lowutil_validation_20261027.md)：TP4、同显存上限下 FP8/BF16 的 block 容量、短请求和 8192-token 精度对照。
- [历史临时接口测试说明](legacy_g128_test_guide.md)：从kernel仓README提取，命令已更新到归档位置。
- [测试和补丁入口](../../work-tests/fp8_kvcache/README.md)：CPU golden、旧设备测试、增强版harness快照和撤回源码的补丁。

本次归档后，DLC_Custom_Kernel中的研究文档、新测试脚本和两处tracked测试修改均已移出/恢复；PyTorch临时 `dlc_dequant_e4m3` wrapper、native schema和torchgen例外均恢复至当前HEAD。未修改其他已有源码改动，未重编译或卸载PyTorch。因此已编译库仍可能暴露临时接口，下次按恢复后的源码构建后才会消失。

归档期间只调整了文档入口、离线HTML下载链接和测试命令路径。六份issue正文保持自包含；历史JSON和PDF不重写。原设计中的旧机器源码路径保留为历史记录，需要查实现时在当前工作区按相应符号定位。
