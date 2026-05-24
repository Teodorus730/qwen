# Real sample readiness checklist

## Before first HF run

- [ ] User explicitly allows network/HF.
- [ ] Exact dataset id verified.
- [ ] Split/config verified.
- [ ] `max_docs` chosen.
- [ ] Output directory chosen.
- [ ] Disk space roughly checked.
- [ ] Sample size is tiny.
- [ ] No full dataset download.
- [ ] Commit policy decided.
- [ ] Labels to run selected.
- [ ] Expected runtime accepted.

## After first HF run

- [ ] Validate chunks.
- [ ] Inspect token length distribution.
- [ ] Inspect source_type distribution.
- [ ] Run rule-based labels.
- [ ] Run lexical labels.
- [ ] Compare label runs.
- [ ] Manually review 20 examples.
- [ ] Decide whether taxonomy needs changes.
- [ ] Do not jump to NLL scoring until labels look sane.
