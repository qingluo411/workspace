# vLLM DLC Proactive Upgrade Workflow

## 1. Confirm paths and target

Resolve and confirm:

- vLLM root or matching worktree. Prefer the user's path; common local names are `vllm` and `vllm-main`.
- vllm-dlc root. Prefer the user's path; common local names are `vllm-dlc` and `vllm-dlc-main`.
- Target vLLM commit, either provided by the user or read from the local vLLM repository.
- Target vLLM version. If the target commit is exactly tagged, use that tag. If not, search backward through the target commit's history and use the nearest ancestor version tag.

Use the local vLLM repository to resolve the version label:

```bash
# Exact tag on the target commit, if one exists.
git tag --points-at <NEW_COMMIT> | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+'

# Fallback: nearest ancestor version tag.
git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' <NEW_COMMIT>
```

If multiple exact tags exist, prefer the stable version tag over rc/dev tags unless the user requested an rc/dev target.

## 2. Prepare the matching vLLM target

Inspect all registered worktrees and the target worktree's status before changing branches. If the user has not already prepared the target worktree, create it without disturbing the primary vLLM checkout.

Fetch both remotes before resolving the final target:

```bash
git fetch origin
git fetch upstream
```

Some DLC environments maintain compatibility fixes on `origin/main` ahead of upstream. When that is the repository's intended integration model, rebase the origin mainline onto the latest upstream mainline in the matching worktree, then use the rebased commit—not raw `upstream/main`—as the vllm-dlc alignment target. Check for already-landed compatibility fixes before recreating them in the plugin.

Never remove a worktree merely because the directory was deleted manually. Inspect `git worktree list` and prune/remove only the exact stale registration requested by the user.

## 3. Check git safety before branch operations

From the vllm-dlc root, inspect branch and status:

```bash
git branch --show-current
git status --short
```

If there are uncommitted changes, ask the user before switching or creating the target branch. Do not discard changes.

The target vllm-dlc branch name must match the vLLM version exactly. Example: target vLLM `v0.13.0` uses vllm-dlc branch `v0.13.0`.

Check both repositories. If there are uncommitted changes, do not switch, rebase, or recreate branches until the changes are understood and preserved.

## 4. Read current vLLM alignment

Read `vllm_cl/vllm_alignment.yaml` to find the old vLLM version and commit.

Expected schema:

```yaml
vllm:
  version: v0.13.0
  commit: <full_sha>
vllm_cl:
  branch: v0.13.0
  updated_at: 2026-05-13
```

## 5. Compare upstream vLLM changes

Run comparison in the local vLLM repository:

```bash
git log --oneline <OLD_COMMIT>..<NEW_COMMIT>
git diff <OLD_COMMIT> <NEW_COMMIT> --name-status
git diff <OLD_COMMIT> <NEW_COMMIT> -- vllm/
```

Prioritize changes in:

- `vllm/platforms/`
- `vllm/model_executor/layers/attention/`
- `vllm/model_executor/layers/fused_moe/`
- `vllm/model_executor/models/`
- `vllm/distributed/`
- `vllm/v1/worker/`
- `vllm/worker/`
- `vllm/config*`
- `vllm/model_executor/layers/quantization/`
- `vllm/model_executor/custom_op.py`
- KV cache, sampling, and logits processing paths
- V2 model runner, attention metadata builders, and distributed communicator state

## 6. Use the patch import manifest

Read `vllm_cl/patch_import_manifest.yaml`. For every changed vLLM file, check whether it maps to a listed module or watched symbol.

If a watched upstream class, function, forward signature, config field, or behavior changed, update the corresponding vllm-dlc patch. Simple import existence is not enough.

## 7. Apply minimal DLC adaptations

Adapt only the files needed for the changed upstream behavior. Prefer direct updates to existing patch files. Do not add compatibility branches for many vLLM versions unless the user explicitly requests them; vllm-dlc uses one branch per vLLM version.

Preserve DLC constraints:

- **every startup command must include `--block-size 256`**
- no Triton kernels
- no `torch.compile` or `torch._dynamo` compiled paths
- use PyTorch native or DLC-supported operators for unsupported paths
- do not advertise or select a dtype/quantization layout merely because an import-time alias makes the symbol exist
- preserve DLC KV-cache physical layout, offset, and padding invariants from `references/runtime-validation.md`

## 8. Run layered validation

Read `references/runtime-validation.md` and validate in layers:

- source-level/unit tests for changed patch families
- DLC operator CI when kernels or collectives changed
- standard infer CI and TYD infer CI for supported models
- targeted real-weight smoke tests for Llama, DeepSeek-V2, Qwen3, and GLM-5.1 as applicable
- the same representative smoke matrix with `VLLM_USE_V2_MODEL_RUNNER=1`
- KV-cache runtime invariants and capacity at the production memory-utilization setting

Do not use a fixed one-minute startup timeout. Large models and CI batches can spend several minutes loading their first model. Continue while shard progress, logs, device activity, or process state shows forward progress. Treat a run as stuck only after repeated unchanged observations and inspection of the parent plus per-rank logs.

For server-mode tests, still require:

```bash
curl -s http://127.0.0.1:<PORT>/v1/models

curl -s http://127.0.0.1:<PORT>/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<SERVED_MODEL_NAME>","prompt":"Hello","max_tokens":5,"temperature":0}'

curl -s http://127.0.0.1:<PORT>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<SERVED_MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":5,"temperature":0}'
```

Pass requires HTTP 200, non-empty generated text, and the server process must
remain alive after both short request paths and the long-prefix request. The
long-prefix request should use enough prompt tokens to trigger chunked prefill
for the configured `max_model_len` while staying within the model context limit.
Startup-only validation is not enough.

Use a model-appropriate prompt. A generic `Hello` can legitimately emit EOS immediately on some models; inspect token IDs and finish reason, then use the repository CI prompt/golden prefix before classifying an empty string as a model failure. `ignore_eos=True` is diagnostic only and is not itself a pass gate.

Dummy validation is not an acceptable substitute for main2main completion. If a required model path is unavailable, stop and ask the user to fix the environment or explicitly change the gate.

## 9. Update metadata

After successful adaptation and validation:

- update `vllm_cl/vllm_alignment.yaml` with the new vLLM version, new vLLM commit, current vllm-dlc branch, and current date
- update `vllm_cl/patch_import_manifest.yaml` if patch dependencies changed

## 10. Final summary

Respond in compact Chinese with:

- old and new vLLM commits
- vllm-dlc branch
- changed files
- upstream changes adapted
- patch manifest updates
- source/unit, operator, infer, and TYD CI results
- default and V2 real-weight model results
- KV-cache layout, storage offset, padding-aware capacity, and copy/profile evidence
- skipped tests or expected failures, with the exact reason and scope
- known risks
