# Versioned full-baseline results

No full baseline has been recorded yet. The HellaSwag run with `--limit 10`
is a smoke test only and must not be entered here as a baseline score.

After a reviewed full run, retain the compact JSON written by
`--write-baseline-summary` under `baseline_results/` and add a short table
here with its run ID, model revision, backend/dtype, lm-eval version and the
five task metrics. The adjacent `.environment.txt` records `pip freeze` for
the same run. Raw harness JSON and sample logs remain in `results/` and are
intentionally ignored.
