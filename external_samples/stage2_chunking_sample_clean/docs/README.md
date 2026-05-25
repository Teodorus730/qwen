# Stage2 documentation index

This folder contains stage2 documentation for corpus chunking and domain labeling.

Read first:

1. `current_status_ru.md` - Russian handoff and current status.
2. `source_status_ru.md` - dataset/source status.
3. `pipeline_map_ru.md` - pipeline inputs, outputs, and scripts.
4. `validation_modes_ru.md` - how validation differs for synthetic and real samples.
5. `classifier_contract_ru.md` - classifier output contract before MiniLM.
6. `data_policy_ru.md` - tracked vs temporary artifacts.
7. `minilm_readiness_plan_ru.md` - MiniLM plan without model download.

Audit reports:

- `repo_audit/01_structure_organization_problems.md`
- `repo_audit/02_documentation_language_problems.md`
- `repo_audit/03_code_pipeline_problems.md`
- `repo_audit/04_data_artifacts_git_hygiene_problems.md`
- `repo_audit/05_final_audit_summary_and_minilm_readiness.md`
- `repo_audit/06_branch_scope_and_safe_boundaries.md`
- `repo_audit/07_cleanup_stabilization_log.md`

Historical or detailed docs remain useful as appendix material, but they are not the first entry point for the team.

Current scope:

- chunking;
- source metadata;
- domain labeling;
- rule-based and lexical baselines;
- tiny real samples FineWeb-Edu and FineMath;
- preparation for MiniLM/Sentence-Transformers nearest-label classifier.

Downstream/out of current scope:

- NLL/logprob scoring;
- effective context window;
- model training.
