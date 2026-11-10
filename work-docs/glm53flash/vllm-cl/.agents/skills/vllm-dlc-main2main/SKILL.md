---
name: vllm-dlc-main2main
description: "Use when upgrading vllm-dlc to a newer upstream vLLM version or commit, creating a matching vllm-dlc version branch, or analyzing whether upstream vLLM changes require DLC patch updates and runtime validation."
---

# vLLM DLC main2main

## Overview

Adapt vllm-dlc to upstream vLLM changes by comparing the currently tracked vLLM commit with a target vLLM version or commit, updating DLC patches, and validating with local real-weight smoke tests.

## Read order

Do not read reference files before collecting the target. Start with the Initial response rule below.

After the target vLLM version or commit is known:

1. Read `proactive-upgrade.md`.
2. Read `references/patch-import-manifest.md` before analyzing changed vLLM model files or vllm-dlc patch files.
3. Read `references/runtime-validation.md` before changing KV-cache, dtype, quantization, runner, or CI behavior, and always before final validation.

If a reference file cannot be read because of a tool parameter issue, do not mention the tool failure to the user. Retry the read without optional PDF-only parameters, or continue from the workflow summary in this file and read the reference later.

## When to use

Use this skill when:

- The user wants vllm-dlc to follow a newer vLLM version or commit.
- The user wants to create a vllm-dlc branch corresponding to a vLLM version.
- The user asks whether upstream vLLM changes require vllm-dlc patch updates.
- The user asks to analyze the impact of changed vLLM model files on `vllm_cl/patch`.

## Initial response rule

If the user did not provide a target vLLM version or commit, ask only this question:

> Which upstream vLLM version or commit should vllm-dlc be adapted to?

Do not report internal skill-loading, reference-reading, or tool-parameter issues while asking for the target. The user only needs to provide the target.

## Branch policy

- Resolve the target vLLM version from the target commit before branch operations.
- If the target commit is exactly on a vLLM version tag, use that tag.
- If the target commit is not exactly tagged, search backward through history and use the nearest ancestor version tag.
- vllm-dlc branch names match the resolved vLLM version exactly.
- Example: vLLM `v0.13.0` maps to vllm-dlc branch `v0.13.0`.
- Before switching or creating a branch, show the current branch and `git status`.
- If uncommitted changes exist, ask the user before branch operations.
- Never discard user changes with reset, checkout, clean, or similar destructive commands.

## Workflow summary

After the target is known:

1. Resolve the vLLM and vllm-dlc roots; do not assume a fixed workspace prefix.
2. Prepare a clean matching vLLM worktree. Fetch both `origin` and `upstream`; when the maintained target includes origin compatibility commits, rebase `origin/main` onto `upstream/main` and use the resulting commit as the target.
3. Resolve the target vLLM version from the final target commit.
4. Check both repositories' branches, worktree registration, and `git status` before branch operations.
5. Read `vllm_cl/vllm_alignment.yaml` for the old vLLM commit.
6. Compare `old..new` commits and vLLM file diffs.
7. Use `vllm_cl/patch_import_manifest.yaml` to locate affected DLC patches.
8. Apply minimal DLC updates, including KV-cache physical-layout and supported-dtype audits.
9. Run source tests, DLC CI/TYD gates, and the default/V2 real-weight matrix described in `references/runtime-validation.md`.
10. Update alignment and patch manifest metadata.

## Output contract

The skill produces:

- code changes in the working tree
- updated `vllm_cl/vllm_alignment.yaml`
- updated `vllm_cl/patch_import_manifest.yaml` when dependencies change
- KV-cache layout, dtype, stride, storage-offset, physical-padding, and capacity evidence
- CI, TYD, default-runner, and V2-runner validation evidence, with any scoped expected failures called out
- compact Chinese summary with commit range, files changed, validation results, model paths, and known risks

The skill does not create a commit unless the user explicitly asks.
