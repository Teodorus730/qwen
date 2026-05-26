# MiniLM dependency strategy

Date: 2026-05-27

Scope: decide how to prepare local dependencies for the optional MiniLM/Sentence-Transformers embedding baseline without installing anything yet.

No packages were installed, no models were downloaded, no HF streaming was run, and no MiniLM inference was started while preparing this document.

## Current status

Current Python executable:

```text
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe
```

Current Python version:

```text
Python 3.14.1
```

Local Python inventory:

- `.venv` uses Python 3.14.1.
- `where python` finds:
  - `C:\Users\pervo\AppData\Local\Microsoft\WindowsApps\python.exe`;
  - `C:\Users\pervo\AppData\Local\Python\bin\python.exe`.
- `C:\Users\pervo\AppData\Local\Python\bin\python.exe` is also Python 3.14.1.
- `py -0p` failed through the WindowsApps launcher, so it did not provide a reliable interpreter list.
- No local Python 3.10, 3.11, or 3.12 executable was found in the checked paths.

Missing packages in current `.venv`:

- `sentence_transformers`;
- `torch`;
- `transformers`;
- `huggingface_hub`;
- `numpy`.

Local model cache status:

- `C:\Users\pervo\.cache\huggingface\hub` exists.
- `C:\Users\pervo\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2` was not found.
- `C:\Users\pervo\.cache\torch\sentence_transformers` was not found.
- `C:\Users\pervo\.cache\sentence_transformers` was not found.

## Why real MiniLM run is blocked

The embedding classifier script is ready for dry-run and fail-closed local execution, but real inference needs packages and model files that are not currently available.

Blocked because:

- `sentence-transformers` is missing;
- `torch` is missing;
- `numpy` is missing;
- the MiniLM model cache is missing;
- no approved local model path is recorded;
- installing dependencies or downloading a model requires explicit approval.

The current script should not silently download a model because real runs use `local_files_only=True`, but the environment must be prepared before a real run can succeed.

## Main options

### Option A: use current project .venv

Pros:

- simple;
- same repo environment;
- shortest commands;
- no extra interpreter selection.

Cons:

- may risk mixing experimental ML dependencies with lightweight stage2 tools;
- Python 3.14.1 compatibility must be checked before install;
- PyTorch/Sentence-Transformers wheels may lag newest Python versions;
- failed installs could leave the existing `.venv` messy.

### Option B: create separate embedding venv

Pros:

- safer isolation;
- easier to delete/recreate;
- avoids polluting current `.venv`;
- keeps no-dependency rule-based/lexical stage2 checks lightweight;
- makes it easier to test a Python version better supported by PyTorch if one is installed later.

Cons:

- extra setup;
- commands are longer;
- currently no local Python 3.10/3.11/3.12 was found, so a better interpreter may need to be installed or selected later by the user;
- output paths must be run carefully from the stage2 folder.

### Option C: use Colab/Kaggle later

Pros:

- easier model/dependency setup;
- more standard ML environment;
- possible GPU access;
- less local disk/cache friction.

Cons:

- less local reproducibility;
- not ideal before local baseline is stable;
- requires moving data/code or reproducing stage2 paths;
- harder to keep strict no-download/local-only behavior identical to the local repo workflow.

## Recommended option

Recommended: Option B, a separate embedding venv.

Reason:

- MiniLM/Sentence-Transformers adds heavier ML dependencies than the current rule-based and lexical stage2 tools need.
- The current `.venv` is Python 3.14.1, which may be risky for PyTorch/Sentence-Transformers compatibility.
- Keeping embedding experiments isolated reduces the chance of breaking the already stable local MVP pipeline.
- If a Python 3.10/3.11/3.12 interpreter becomes available, the separate embedding venv can use it without touching the current `.venv`.

If the team wants the fastest path and accepts dependency risk, Option A is possible after explicit approval, but it is not the safest default.

## Exact future setup commands

Do not run these commands until explicit approval is given.

### Variant 1: current .venv, only after explicit approval

From repo root:

```powershell
cd C:\Users\pervo\PycharmProjects\qwen
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe -m pip install -r external_samples\stage2_chunking_sample_clean\requirements-embedding.txt
```

Then verify imports:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe -c "import sentence_transformers, torch, numpy; print(sentence_transformers.__version__); print(torch.__version__); print(numpy.__version__)"
```

### Variant 2: separate .venv, only after explicit approval

If a compatible Python is available later, prefer Python 3.10, 3.11, or 3.12 for the embedding venv.

Example using an explicit Python path placeholder:

```powershell
cd C:\Users\pervo\PycharmProjects\qwen
<PYTHON_3_10_11_OR_12_EXE> -m venv .venv-embedding
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe -m pip install -r external_samples\stage2_chunking_sample_clean\requirements-embedding.txt
```

If only the current Python 3.14.1 is available and the team approves testing it:

```powershell
cd C:\Users\pervo\PycharmProjects\qwen
C:\Users\pervo\AppData\Local\Python\bin\python.exe -m venv .venv-embedding
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe -m pip install -r external_samples\stage2_chunking_sample_clean\requirements-embedding.txt
```

Then verify imports:

```powershell
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe -c "import sentence_transformers, torch, numpy; print(sentence_transformers.__version__); print(torch.__version__); print(numpy.__version__)"
```

Version pins are intentionally not added yet. Pin versions after choosing the Python version and confirming which PyTorch/Sentence-Transformers wheels resolve cleanly.

## Exact future MiniLM run command

Recommended if using separate `.venv-embedding`:

```powershell
cd C:\Users\pervo\PycharmProjects\qwen\external_samples\stage2_chunking_sample_clean
C:\Users\pervo\PycharmProjects\qwen\.venv-embedding\Scripts\python.exe scripts\classify_chunks_embedding_baseline.py `
  --input data_samples\classifier_benchmark_chunks.jsonl `
  --labels taxonomy\simple_domain_labels.json `
  --output data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --batch-size 32 `
  --top-k 3
```

If using current `.venv` after approval:

```powershell
cd C:\Users\pervo\PycharmProjects\qwen\external_samples\stage2_chunking_sample_clean
C:\Users\pervo\PycharmProjects\qwen\.venv\Scripts\python.exe scripts\classify_chunks_embedding_baseline.py `
  --input data_samples\classifier_benchmark_chunks.jsonl `
  --labels taxonomy\simple_domain_labels.json `
  --output data_samples\classifier_benchmark_minilm_labeled.jsonl `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --batch-size 32 `
  --top-k 3
```

Because the script requests `local_files_only=True`, this run requires the model to already be available locally. If the model is missing, it should fail closed.

## Safety rules

- Install packages only after explicit approval.
- Download models only after explicit approval.
- Do not run HF streaming during MiniLM setup.
- First real MiniLM run must be on the synthetic benchmark only.
- Do not overwrite rule-based or lexical outputs.
- Validate MiniLM output before comparing it.
- Compare rule-based vs lexical vs MiniLM before moving to real samples.
- Run FineWeb-Edu and FineMath tiny samples only after benchmark behavior is reviewed.
- Keep OpenWebMath optional_later.
- Keep NLL/logprob/effective context out of this MiniLM setup step.
