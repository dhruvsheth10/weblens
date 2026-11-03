"""Score DDA and OmniAID on the authorized tuning rows and write the report."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.image_io import decode_image
from research.reddit_adaptation.common import safe_data
from research.reviewers.baselines import aligned, load_baselines
from research.reviewers.dda_runner import build as build_dda
from research.reviewers.metrics import agreement, evaluate
from research.reviewers.omniaid_runner import build as build_omniaid
from research.reviewers.report import write_report
from research.reviewers.rows import calibration_rows, reddit_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/reviewer-screen-2026-10-16'
BUILDERS = {'dda': build_dda, 'omniaid': build_omniaid}


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        try:
            probe = torch.ones(1, device='mps')
            torch.mps.synchronize()
            del probe
            return torch.device('mps')
        except Exception as error:
            print(f'MPS probe failed: {error}', flush=True)
    return torch.device('cpu')


def public_row(row: dict, variant: str) -> dict:
    return {
        'set': row['set'],
        'variant': variant,
        'id': row['id'],
        'file': row['file'],
        'sourceHash': row['sourceHash'],
        'label': row['label'],
        'category': row['category'],
    }


def decode_variant(row: dict, variant: str) -> Image.Image:
    data = safe_data(row)
    decode_name = 'original' if variant == 'preview' else variant
    image, _size = decode_image(data, decode_name)
    return image


def score(model: str, rows: list[dict], limit: int | None) -> tuple[list[dict], dict]:
    device = choose_device()
    print(f'BUILD {model} on {device}', flush=True)
    predict, info = BUILDERS[model](device)
    destination = OUT / (f'{model}-smoke.jsonl' if limit else f'{model}-predictions.jsonl')
    done = {}
    if destination.exists() and not limit:
        for line in destination.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if 'logit' in record:
                done[(record['set'], record['variant'], record['sourceHash'])] = record
    jobs = []
    for row in rows:
        for variant in row['variants']:
            key = (row['set'], variant, row['sourceHash'])
            if key not in done:
                jobs.append((row, variant))
    if limit is not None:
        jobs = jobs[:limit]
        done = {}
    print(f'SCORE {model} pending={len(jobs)} resumed={len(done)}', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if limit else 'a'
    fallen_back = False
    with destination.open(mode) as handle:
        for index, (row, variant) in enumerate(jobs, start=1):
            record = public_row(row, variant)
            try:
                image = decode_variant(row, variant)
                try:
                    logit, elapsed = predict(image)
                except Exception as error:
                    if device.type != 'mps' or fallen_back:
                        raise
                    print(f'MPS forward failed, rebuilding on CPU: {error}', flush=True)
                    device = torch.device('cpu')
                    predict, info = BUILDERS[model](device)
                    info['mpsError'] = f'{type(error).__name__}: {error}'
                    fallen_back = True
                    logit, elapsed = predict(image)
                record['logit'] = logit
                record['elapsedMs'] = elapsed
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                print('FAIL', record['file'], variant, record['error'], flush=True)
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            if index <= 3 or index % 25 == 0 or index == len(jobs):
                shown = record.get('elapsedMs')
                print(f'{model} {index}/{len(jobs)} {record["set"]} {variant} {shown}', flush=True)
    with destination.open() as handle:
        loaded = [json.loads(line) for line in handle if line.strip()]
    latest = {}
    for record in loaded:
        key = (record['set'], record['variant'], record['sourceHash'])
        previous = latest.get(key)
        if previous is None or 'logit' in record:
            latest[key] = record
    records = list(latest.values())
    scored = [record for record in records if 'logit' in record]
    info['scored'] = len(scored)
    info['errors'] = len(records) - len(scored)
    info['predictions'] = str(destination)
    return records, info


def latency_summary(records: list[dict]) -> dict | None:
    times = [record['elapsedMs'] for record in records if 'elapsedMs' in record]
    if not times:
        return None
    values = np.asarray(times, dtype=float)
    return {
        'n': int(values.size),
        'meanMs': float(values.mean()),
        'medianMs': float(np.median(values)),
        'p90Ms': float(np.percentile(values, 90)),
        'firstMs': float(values[0]),
    }


def metrics_for(records: list[dict], baselines: dict) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if 'logit' not in record:
            continue
        grouped.setdefault((record['set'], record['variant']), []).append(record)
    output = {}
    for (split, variant), group in sorted(grouped.items()):
        rows = [{key: record[key] for key in ('label', 'category', 'sourceHash', 'id', 'file')} for record in group]
        logits = np.array([record['logit'] for record in group], dtype=float)
        block = {'candidate': evaluate(logits, rows), 'latency': latency_summary(group), 'baselines': {}, 'agreement': {}}
        stored = baselines['logits'].get(split, {}).get(variant, {})
        by_hash = {record['sourceHash']: record['logit'] for record in group}
        for name in ('MIRROR', 'Community+Ferret', 'reddit-student-last2-v1'):
            mapping = stored.get(name) or {}
            kept, baseline_logits = aligned(rows, mapping)
            if not kept:
                block['baselines'][name] = {'available': False, 'reason': 'No stored logits for this set and variant'}
                continue
            candidate_logits = np.array([by_hash[row['sourceHash']] for row in kept], dtype=float)
            block['baselines'][name] = {
                'available': True,
                'intersection': len(kept),
                'candidateOnIntersection': evaluate(candidate_logits, kept),
                'baseline': evaluate(baseline_logits, kept),
            }
            if name == 'MIRROR':
                block['agreement'] = agreement(candidate_logits, baseline_logits, kept)
        output[f'{split}/{variant}'] = block
    return output


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def disk_usage() -> dict:
    third = ROOT / 'research/reviewers/third_party'
    output = subprocess.check_output(['du', '-sk', str(third), str(OUT)], text=True)
    sizes = {}
    for line in output.splitlines():
        kib, path = line.split('\t', 1)
        sizes[path] = int(kib) * 1024
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dda', 'omniaid', 'baselines', 'report'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    calibration = calibration_rows()
    reddit, reddit_summary = reddit_rows()
    (OUT / 'row-summary.json').write_text(json.dumps({
        'calibrationTuning': len(calibration),
        'reddit': {key: value for key, value in reddit_summary.items() if key != 'skipReasons'},
        'redditSkipReasons': reddit_summary['skipReasons'],
    }, indent=2) + '\n')
    baselines = load_baselines(calibration, reddit)
    (OUT / 'baselines.json').write_text(json.dumps(baselines, indent=2) + '\n')
    print(json.dumps(baselines['checks'], indent=2), flush=True)
    if args.model == 'baselines':
        return
    rows = calibration + reddit
    metadata_path = OUT / 'run-metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {'models': {}}
    metadata['revisions'] = {
        'dda': git_revision(ROOT / 'research/reviewers/third_party/Dual-Data-Alignment'),
        'omniaid': git_revision(ROOT / 'research/reviewers/third_party/OmniAID'),
    }
    metadata['torch'] = torch.__version__
    metadata['mpsAvailable'] = bool(torch.backends.mps.is_available())
    if args.model in BUILDERS:
        started = time.perf_counter()
        try:
            records, info = score(args.model, rows, args.limit)
        except Exception:
            traceback.print_exc()
            metadata['models'][args.model] = {'ran': False, 'error': traceback.format_exc()}
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            raise
        if args.limit:
            print(json.dumps({'smoke': records, 'info': info}, indent=2, default=str)[:4000])
            return
        prediction_path = OUT / f'{args.model}-predictions.json'
        prediction_path.write_text(json.dumps({'info': info, 'records': records}, indent=2) + '\n')
        metadata['models'][args.model] = {
            'ran': info.get('errors', 1) < len(records) and any('logit' in record for record in records),
            'info': info,
            'seconds': time.perf_counter() - started,
            'metrics': metrics_for(records, baselines),
        }
    metadata['diskBytes'] = disk_usage()
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    write_report(OUT, metadata, reddit_summary)


if __name__ == '__main__':
    main()
