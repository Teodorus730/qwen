"""Frozen independent-encoder benchmark: data, exact JS and verified exports."""
from pathlib import Path
import base64
from collections import Counter
import hashlib
import io
import json
import math
import re
import time
import zipfile

ROOT = Path('/content/gp_benchmark_20261001')
CFG = json.loads((ROOT / 'protocol.json').read_text())
MODELS = {}
TOKENIZERS = {}
MODEL_SPECS = {}
TOKENIZER_SPECS = {}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_json(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)
    return path


def load_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]


def save_rows(name, rows):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return path


def export(names, job_id):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    raw = buffer.getvalue()
    print('GP_ARTIFACT=' + json.dumps({'job_id': job_id, 'sha256': hashlib.sha256(raw).hexdigest(),
          'zip_base64': base64.b64encode(raw).decode()}), flush=True)


def tokenizer(role):
    from transformers import AutoTokenizer
    spec = CFG['models'][role]
    if role not in TOKENIZERS or TOKENIZER_SPECS.get(role) != spec:
        TOKENIZERS[role] = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
        TOKENIZER_SPECS[role] = dict(spec)
    return TOKENIZERS[role]


def model(role):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM
    spec = CFG['models'][role]
    if role not in MODELS or MODEL_SPECS.get(role) != spec:
        factory = AutoModel if role == 'encoder' else AutoModelForCausalLM
        MODELS[role] = factory.from_pretrained(spec['id'], revision=spec['revision'],
                            dtype=torch.float32 if role == 'encoder' else torch.float16,
                            attn_implementation='sdpa').to('cuda').eval()
        MODEL_SPECS[role] = dict(spec)
    return MODELS[role]


def clean_reason(text):
    if '\ufffd' in text:
        return 'replacement_character'
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        return 'control_character'
    if len(re.findall(r'(?i)\bread more\b', text)) > 3:
        return 'repeated_read_more'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 12 and sum(bool(re.fullmatch(r'[\d\s.,%+\-=/()]+', line)) for line in lines) / len(lines) > .6:
        return 'numeric_lines'
    if len(lines) >= 20 and sum(len(line) < 15 for line in lines) / len(lines) > .7:
        return 'short_lines'
    if len(lines) >= 20 and max(Counter(lines).values()) / len(lines) > .3:
        return 'repeated_lines'
    return None


def natural_window(text, key):
    # Window coordinates use the independent encoder, never Qwen hidden states.
    tok = tokenizer('encoder')
    ids = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    offsets = ids['offset_mapping']
    if len(offsets) < CFG['natural_min_tokens']:
        return None
    limit = CFG['encoder_max_length'] - 2
    start_token = int(fingerprint(['window', key]), 16) % (max(len(offsets)-limit, 0)+1)
    end_token = min(start_token + limit, len(offsets))
    lo, hi = offsets[start_token][0], offsets[end_token-1][1]
    boundaries = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+|\n{2,}', text)] + [len(text)]
    start_candidates = [p for p in boundaries if lo <= p <= lo+160 and p < hi]
    end_candidates = [p for p in boundaries if hi-160 <= p <= hi and p > lo]
    trim_lo = min(start_candidates) if start_candidates else lo
    trim_hi = max(end_candidates) if end_candidates else hi
    if trim_hi > trim_lo and len(tok(text[trim_lo:trim_hi], add_special_tokens=False)['input_ids']) >= CFG['natural_min_tokens']:
        lo, hi = trim_lo, trim_hi
    window = text[lo:hi].strip()
    count = len(tok(window, add_special_tokens=False)['input_ids'])
    if count < CFG['natural_min_tokens']:
        return None
    return window, lo, hi, count


def controlled(samples_per_domain=100):
    from datasets import load_dataset
    rows = []
    for domain in CFG['mmlu_domains']:
        ds = load_dataset('cais/mmlu', domain, revision=CFG['mmlu_revision'], split='test', streaming=True)
        candidates = []
        for index, item in enumerate(ds):
            text = '\n'.join([item['question'].strip()] + [f'{chr(65+j)}. {str(choice).strip()}' for j, choice in enumerate(item['choices'])] + ['Answer:'])
            candidates.append({'sample_id': fingerprint(['mmlu', CFG['mmlu_revision'], domain, index]),
                 'domain': domain, 'source': 'cais/mmlu', 'revision': CFG['mmlu_revision'], 'source_row': index,
                 'text': text, 'text_hash': fingerprint(re.sub(r'\s+', ' ', text).strip())})
        selected = sorted(candidates, key=lambda row: fingerprint(['controlled', CFG['data_seed'], row['sample_id']]))[:samples_per_domain]
        if len(selected) != samples_per_domain:
            raise ValueError(f'{domain}: requested {samples_per_domain}, available {len(candidates)} MMLU test texts')
        rows.extend(selected)
    save_rows('data/controlled.jsonl', rows)
    return rows


def natural_population(source):
    from datasets import load_dataset
    spec = CFG['natural_sources'][source]
    other_hashes = set()
    for path in sorted((ROOT / 'data').glob('population_*.jsonl')):
        if path.name != 'population_' + source + '.jsonl':
            other_hashes.update(row['document_hash'] for row in load_rows(str(path.relative_to(ROOT))))
    stream = load_dataset(spec['dataset'], spec['config'], revision=spec['revision'], split='train', streaming=True)
    stream = stream.shuffle(seed=CFG['data_seed'], buffer_size=CFG['shuffle_buffer'])
    records, seen_docs, seen_windows, rejected = [], set(other_hashes), set(), Counter()
    scanned = 0
    for index, row in enumerate(stream):
        scanned += 1
        text = str(row.get('text', ''))
        reason = clean_reason(text)
        if reason:
            rejected[reason] += 1; continue
        doc_hash = fingerprint(re.sub(r'\s+', ' ', text).strip())
        if doc_hash in seen_docs:
            rejected['duplicate_document'] += 1; continue
        source_id = str(row.get('id') or row.get('url') or ('stream_row=' + str(index)))
        # Explicitly bounded region avoids tokenizing arbitrarily large webpages.
        origin = 0
        if len(text) > CFG['region_chars']:
            origin = int(fingerprint(['region', source, doc_hash]), 16) % (len(text)-CFG['region_chars']+1)
        region = text[origin:origin+CFG['region_chars']]
        window = natural_window(region, [source, doc_hash])
        if window is None:
            rejected['short_window'] += 1; continue
        visible, lo, hi, length = window
        window_hash = fingerprint(re.sub(r'\s+', ' ', visible).strip())
        if window_hash in seen_windows:
            rejected['duplicate_window'] += 1; continue
        seen_docs.add(doc_hash); seen_windows.add(window_hash)
        records.append({'sample_id': fingerprint([source, spec['revision'], doc_hash, origin+lo, origin+hi]),
              'domain': source, 'source': spec['dataset'], 'config': spec['config'], 'revision': spec['revision'],
              'source_id': source_id, 'document_hash': doc_hash, 'text_hash': window_hash, 'text': visible,
              'source_num_chars': len(text), 'start_char': origin+lo, 'end_char': origin+hi,
              'encoder_tokens_without_special': length, 'stream_row': index})
        if len(records) >= CFG['natural_population'] or scanned >= CFG['scan_limit']:
            break
    assert len(records) >= CFG['natural_per_fold'] * 3, (source, scanned, len(records))
    save_rows('data/population_' + source + '.jsonl', records)
    report = {'source': source, 'scanned': scanned, 'accepted': len(records), 'rejected': dict(rejected),
              'config_hash': fingerprint(CFG), 'scope': 'Fixed bounded stream population, not representative of entire source'}
    save_json('data/audit_' + source + '.json', report)
    return report


def freeze_natural():
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.neighbors import NearestNeighbors
    rows = []
    for source in CFG['natural_sources']:
        rows += load_rows('data/population_' + source + '.jsonl')
    vectors = HashingVectorizer(analyzer='char', ngram_range=(5, 5), n_features=2**18,
                               alternate_sign=False, dtype=np.float32).transform([row['text'] for row in rows])
    nn = NearestNeighbors(n_neighbors=8, metric='cosine', algorithm='brute', n_jobs=2).fit(vectors)
    distances, neighbors = nn.kneighbors(vectors)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    pairs = []
    for i in range(len(rows)):
        for distance, j in zip(distances[i], neighbors[i]):
            if j != i and distance <= 1-CFG['near_duplicate_cosine']:
                a, b = find(i), find(int(j))
                parent[max(a,b)] = min(a,b)
                if j > i: pairs.append({'a': rows[i]['sample_id'], 'b': rows[int(j)]['sample_id'], 'cosine': float(1-distance)})
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(find(i), []).append(row)
    representatives = [min(group, key=lambda row: fingerprint(['representative', row['sample_id']])) for group in groups.values()]
    for source in CFG['natural_sources']:
        ordered = sorted([r for r in representatives if r['domain'] == source], key=lambda row: fingerprint(['fold_order', row['sample_id']]))
        assert len(ordered) >= 3*CFG['natural_per_fold'], (source, len(ordered))
        for fold in range(3):
            selected = ordered[fold*CFG['natural_per_fold']:(fold+1)*CFG['natural_per_fold']]
            save_rows(f'data/{source}_fold{fold}.jsonl', [dict(row, dataset_fold=fold) for row in selected])
    save_json('data/near_duplicates.json', {'threshold': CFG['near_duplicate_cosine'], 'pairs': pairs,
              'candidate_documents': len(rows), 'representatives': len(representatives),
              'detector_scope': '8 nearest char5 neighbors; not proof of all semantic duplicates'})
    for fold in range(3):
        selected = []
        for source in CFG['natural_sources']: selected += load_rows(f'data/{source}_fold{fold}.jsonl')
        save_rows(f'data/natural_fold{fold}.jsonl', selected)


def metric_from_logits(a, b):
    import torch
    logp, logq = a.float().log_softmax(-1), b.float().log_softmax(-1)
    logm = torch.logaddexp(logp, logq)-math.log(2.)
    return .5*((logp.exp()*(logp-logm)).sum(-1)+(logq.exp()*(logq-logm)).sum(-1))


def metric_exact(a, b, chunk=8):
    import torch
    return torch.cat([metric_from_logits(a[i:i+chunk], b[i:i+chunk]).cpu() for i in range(0, len(a), chunk)]).numpy()


def metric_topk(a, b, k, chunk=16):
    import torch
    values = []
    for start in range(0, len(a), chunk):
        t, s = a[start:start+chunk].float(), b[start:start+chunk].float()
        k = min(k, t.shape[-1])
        it, iq = t.topk(k, dim=-1).indices, s.topk(k, dim=-1).indices
        candidates = torch.cat([it, iq], -1)
        duplicate = (iq.unsqueeze(-1) == it.unsqueeze(-2)).any(-1)
        valid = torch.cat([torch.ones_like(it, dtype=torch.bool), ~duplicate], -1)
        ta = t.gather(-1, candidates).masked_fill(~valid, torch.finfo(torch.float32).min)
        sb = s.gather(-1, candidates).masked_fill(~valid, torch.finfo(torch.float32).min)
        values.append(metric_from_logits(ta, sb).cpu())
    return torch.cat(values).numpy()


def measure(rows, topk=False, deadline_s=360):
    import numpy as np
    import torch
    tok = tokenizer('student'); other = tokenizer('teacher')
    assert tok.get_vocab() == other.get_vocab(), 'Teacher/student vocabulary mismatch'
    started = time.perf_counter()
    completed = []
    for row in rows:
        identity = fingerprint({'sample_id': row['sample_id'], 'text_hash': row['text_hash'],
                'models': {r: CFG['models'][r] for r in ['student', 'teacher']},
                'max_length': CFG['qwen_max_length'], 'positions': 'all_valid_including_last',
                'metric': 'full_vocabulary_js_natural_log_fp32_v1'})
        path = ROOT / 'measurements' / (identity + '.npz')
        if path.exists():
            with np.load(path) as data:
                if not topk or 'top30' in data: completed.append(row['sample_id']); continue
        ids = tok(row['text'], add_special_tokens=False, truncation=True,
                  max_length=CFG['qwen_max_length'], return_offsets_mapping=True)
        teacher_ids = other(row['text'], add_special_tokens=False, truncation=True,
                            max_length=CFG['qwen_max_length'])['input_ids']
        assert ids['input_ids'] == teacher_ids, 'Teacher/student tokenization mismatch'
        inputs = torch.tensor([ids['input_ids']], device='cuda')
        student, teacher = model('student'), model('teacher')
        with torch.inference_mode():
            a = teacher(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0]
            b = student(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0]
            assert a.shape == b.shape
            exact = metric_exact(a, b)
            values = {'exact': exact, 'ids': np.array(ids['input_ids'], dtype=np.int32),
                      'offsets': np.array(ids['offset_mapping'], dtype=np.int32),
                      'sample_id': np.array(row['sample_id']), 'metric_key': np.array(identity)}
            if topk:
                for k in [30, 100, 256]: values['top'+str(k)] = metric_topk(a, b, k)
        assert np.isfinite(exact).all() and exact.min() >= -1e-6 and exact.max() <= math.log(2)+1e-6
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **values)
        completed.append(row['sample_id'])
        del a, b, inputs
        if time.perf_counter()-started > deadline_s: break
    return completed


def measurement_path(row):
    identity = fingerprint({'sample_id': row['sample_id'], 'text_hash': row['text_hash'],
            'models': {r: CFG['models'][r] for r in ['student', 'teacher']},
            'max_length': CFG['qwen_max_length'], 'positions': 'all_valid_including_last',
            'metric': 'full_vocabulary_js_natural_log_fp32_v1'})
    return ROOT / 'measurements' / (identity + '.npz')


def measurement_table(rows):
    import numpy as np
    result = []
    for row in rows:
        path = measurement_path(row)
        if not path.exists(): continue
        with np.load(path) as data:
            record = {'sample_id': row['sample_id'], 'domain': row['domain'], 'exact': float(data['exact'].mean()),
                      'n_tokens': len(data['exact']), 'metric_key': str(data['metric_key'])}
            for k in [30, 100, 256]:
                if 'top'+str(k) in data: record['top'+str(k)] = float(data['top'+str(k)].mean())
        result.append(record)
    return result
