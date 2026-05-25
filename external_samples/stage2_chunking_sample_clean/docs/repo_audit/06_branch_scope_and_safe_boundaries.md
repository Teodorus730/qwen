# Branch scope and safe boundaries

Date: 2026-05-26

Scope: branch-scope audit before stage2 cleanup work. This file records what the current branch changes relative to `origin/main`, and which areas are safe or protected for future audit fixes. No code changes, HF/network/model runs, commits, pushes, resets, checkouts, cleans, deletions, or `.gitignore` edits were performed in this step.

## Purpose

Before fixing audit findings, we need to avoid turning a stage2 cleanup into a repo-wide cleanup. The earlier audits correctly noted that the active stage2 work is hidden under:

```text
external_samples/stage2_chunking_sample_clean/
```

That hurts onboarding, but it does not mean the immediate fix should be moving stage2 to the repository root or rewriting root-level project structure.

This branch exists to keep stage2 work isolated. The scope audit checks:

- how far the branch is from `origin/main`;
- whether committed changes are stage2-local;
- whether any protected/root areas were touched;
- which areas are safe to edit in the next cleanup iterations.

## Current strategy

Stage2 remains an isolated subproject for now. This is an intentional safety strategy, not only an organizational smell.

Reasons:

- The branch should not disturb shared `main`.
- Other team members may own root docs, notebooks, research files, dataset notes, or meeting notes.
- Stage2 cleanup should not silently rewrite unrelated project history or structure.
- Future MiniLM/Sentence-Transformers work can be prepared inside stage2 first.
- Root-level pointers or moving stage2 can be discussed later with the team.

Immediate cleanup should therefore be stage2-local. The goal is to make stage2 self-contained and understandable inside its current folder.

## Main branch comparison

Baseline used: local `origin/main`.

Fetch status: no fetch was run because `origin/main` exists locally and was sufficient for this audit. Therefore this comparison is based on the current local `origin/main`.

Current branch:

```text
feature/review-stage2-chunking
```

Ahead/behind relative to `origin/main`:

```text
origin/main...HEAD: 0 behind, 32 ahead
```

Committed diff summary:

- 81 files changed relative to `origin/main`.
- 80 files are under `external_samples/stage2_chunking_sample_clean/`.
- 1 file is outside stage2: `.gitignore`.
- Total committed diff stat: 5639 insertions.

Committed directory distribution:

```text
  1.2% external_samples/stage2_chunking_sample_clean/config/
 16.0% external_samples/stage2_chunking_sample_clean/data_samples/real_samples/
 13.5% external_samples/stage2_chunking_sample_clean/data_samples/
 37.0% external_samples/stage2_chunking_sample_clean/docs/
  4.9% external_samples/stage2_chunking_sample_clean/examples/
 19.7% external_samples/stage2_chunking_sample_clean/scripts/
  1.2% external_samples/stage2_chunking_sample_clean/taxonomy/
  4.9% external_samples/stage2_chunking_sample_clean/
```

Main committed changed areas inside stage2:

- stage2 README/check/how-to/team message;
- stage2 docs and reports;
- stage2 scripts for chunking, validation, classification, comparison, planning, and review;
- stage2 examples;
- stage2 config and taxonomy;
- stage2 small data samples and tiny real samples.

Committed changes outside stage2:

- `.gitignore` was modified earlier in the branch to ignore:
  - `__pycache__/`;
  - `*.pyc`;
  - `external_samples/stage2_chunking_sample_clean/data_samples/hf_cache_test/`.

No committed changes relative to `origin/main` were found in the protected project areas:

- root `README.MD`;
- root `scripts/`;
- root `docs/`;
- `research/`;
- `data-filtering/`;
- `datasets/`;
- `distribution/`;
- `results of the meetings/`;
- root notebooks.

Current untracked working-tree items:

- `external_samples/stage2_chunking_sample_clean/docs/repo_audit/` audit files;
- old untracked sweep outputs under `external_samples/stage2_chunking_sample_clean/data_samples/`:
  - `classifier_benchmark_chunks_maxdocs20.jsonl`;
  - `classifier_benchmark_chunks_maxdocs40.jsonl`;
  - `classifier_benchmark_chunks_target120.jsonl`;
  - `classifier_benchmark_labeled_target120.jsonl`;
  - matching `classifier_benchmark_run_stats_*` files.

Current ignored stage2-local temp/cache items:

- `external_samples/stage2_chunking_sample_clean/data_samples/hf_cache_test/`;
- `external_samples/stage2_chunking_sample_clean/scripts/__pycache__/`.

## Safe edit zone

Safe for the next cleanup iterations:

- `external_samples/stage2_chunking_sample_clean/docs/`;
- `external_samples/stage2_chunking_sample_clean/docs/repo_audit/`;
- `external_samples/stage2_chunking_sample_clean/README.md`;
- `external_samples/stage2_chunking_sample_clean/CHECK_HOW_TO.md`;
- `external_samples/stage2_chunking_sample_clean/MESSAGE_TO_TEAM.txt`, if later archived or superseded inside stage2;
- `external_samples/stage2_chunking_sample_clean/config/dataset_sources.json`, for source-status cleanup after explicit decision;
- `external_samples/stage2_chunking_sample_clean/taxonomy/simple_domain_labels.json`, for taxonomy docs/versioning after explicit decision;
- `external_samples/stage2_chunking_sample_clean/data_samples/README.md`, for data policy;
- stage2 docs about data policy, source status, MiniLM readiness, and pipeline maps.

Stage2 scripts are not part of the immediate documentation cleanup. They can be changed later only after a separate implementation decision:

- shared JSONL/taxonomy helpers;
- classifier output contract;
- validation modes;
- MiniLM embedding classifier updates.

Stage2 generated outputs and old sweeps should not be deleted or moved in this current step.

## Protected zones

Do not touch without separate explicit approval:

- root `README.MD`;
- root `scripts/`;
- root `docs/`;
- `research/`;
- `data-filtering/`;
- `datasets/`;
- `distribution/`;
- `results of the meetings/`;
- notebooks;
- root-level reports and PDFs;
- any file outside `external_samples/stage2_chunking_sample_clean/`, unless explicitly approved for stage2 safety.

Special case:

- Root `.gitignore` is already changed in this branch. Do not expand or rewrite it during immediate stage2 cleanup unless explicitly approved.

## Reframing audit finding

Earlier wording:

- Active stage2 work is hidden under `external_samples`.

Corrected interpretation:

- Yes, this is inconvenient for onboarding.
- But right now isolation is useful and intentional.
- Stage2 should remain an obособленный подпроект while we stabilize it.
- The immediate fix is to make stage2 self-contained, well documented, and safe to run from inside its folder.
- A root-level pointer or moving/renaming stage2 is a postponed team-level decision, not an immediate cleanup task.

Recommended immediate wording:

```text
Stage2 is intentionally isolated under external_samples/stage2_chunking_sample_clean for this branch.
Improve onboarding by adding stage2-local docs, status tables, data policy, and pipeline maps.
Do not move stage2 or rewrite root repo without team approval.
```

## Cleanup implication

Can do now, stage2-local:

- add Russian stage2 quickstart/current-state docs;
- add stage2 source status table;
- add stage2 data policy;
- add stage2 MiniLM readiness plan;
- clarify FineMath current MVP math source and OpenWebMath optional_later;
- clarify NLL/effective context window as downstream/out of current scope;
- organize stage2 docs conceptually, without moving protected root files;
- update audit docs to reflect safe boundaries.

Postpone:

- moving stage2 out of `external_samples`;
- changing root README;
- changing root docs;
- changing root scripts;
- reorganizing research/notebooks/meeting notes;
- broad root `.gitignore` cleanup;
- deleting old sweep outputs;
- implementing MiniLM;
- running HF/network/model downloads;
- committing or pushing.

## Commands used

Read-only git commands used for this scope audit:

```bash
git status --short --ignored external_samples\stage2_chunking_sample_clean
git branch --show-current
git rev-parse --verify origin/main
git rev-list --left-right --count origin/main...HEAD
git log --oneline --decorate --graph origin/main..HEAD
git diff --name-status origin/main...HEAD
git diff --dirstat=files,0 origin/main...HEAD
git diff --name-status origin/main...HEAD -- external_samples/stage2_chunking_sample_clean
git diff --name-status origin/main...HEAD -- . ":(exclude)external_samples/stage2_chunking_sample_clean/**"
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- .gitignore
git status --short
```
