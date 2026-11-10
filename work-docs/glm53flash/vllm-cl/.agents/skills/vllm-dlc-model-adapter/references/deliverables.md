# vLLM DLC Model Adapter Deliverables

## Required deliverables

Every completed adaptation must include:

- code changes
- test config or equivalent regression case
- compact Chinese analysis report
- exact startup command
- exact validation requests
- dummy validation evidence if dummy was used
- real-weight validation evidence
- known limitations

## Optional deliverables

Only do these when the user explicitly asks:

- tutorial documentation
- git commit

## Final response template

```markdown
已完成 DLC 模型适配。

变更文件：
- `<path>`：<purpose>

验证结果：
- Dummy：<used/not used>，<result>
- Real weights：<model path>，<result>
- `/v1/models`：<result>
- Chat request：<result>
- Multimodal：<result or not applicable>
- MTP：<tested/skipped by user choice/not applicable>

关键命令：
```bash
<startup command>
<curl command>
```

已知限制：
- <limit or none>
```
```
