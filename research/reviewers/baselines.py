"""Existing per-image logits for MIRROR, Community + Ferret, and the Reddit student.

Community + Ferret's stored component logits are combined with the packaged rule
in backend/community.py. The combination is kept only when it matches stored
combined logits.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from research.reviewers.rows import calibration_rows, reddit_rows

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts'
PROFILE = json.loads((ROOT / 'backend/models/community-ferret-profile.json').read_text())
STUDENT = json.loads((ART / 'reddit-adaptation-2026-10-14/student-profile.json').read_text())
FERRET_THRESHOLD = float(PROFILE['componentThresholds']['ferret'])
FERRET_OFFSET = float(math.log(FERRET_THRESHOLD / (1.0 - FERRET_THRESHOLD)))
STUDENT_OFFSET = float(STUDENT['logitOffset'])


def community_ferret_logit(community: float, ferret: float) -> float:
    return max(community, ferret - FERRET_OFFSET)


def _index(rows: list[dict]) -> dict[str, dict]:
    return {row['sourceHash']: row for row in rows}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _max_abs(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return float(max(abs(left - right) for left, right in pairs))


def load_baselines(calibration: list[dict], reddit: list[dict]) -> dict:
    calibration_hashes = _index(calibration)
    reddit_hashes = _index(reddit)
    mirror_cal = {variant: {} for variant in ('original', 'reddit-webp640')}
    ferret_cal = {variant: {} for variant in ('original', 'reddit-webp640')}
    csv_path = ART / 'mirror-evaluation-2026-10-13/comparison-scores.csv'
    with csv_path.open() as handle:
        for record in csv.DictReader(handle):
            if record['dataset'] != 'local' or record['split'] != 'tuning':
                continue
            if record['sourceHash'] not in calibration_hashes:
                continue
            target = None
            if record['system'] == 'MIRROR':
                target = mirror_cal
            elif record['system'] == 'Community + Ferret':
                target = ferret_cal
            if target is None or record['variant'] not in target:
                continue
            target[record['variant']][record['sourceHash']] = float(record['logit'])

    student_cal = {variant: {} for variant in ('original', 'reddit-webp640')}
    transfer = json.loads((ART / 'reddit-adaptation-2026-10-14/student-calibration-transfer.json').read_text())
    for record in transfer['predictions']:
        if record['sourceHash'] not in calibration_hashes:
            continue
        student_cal[record['variant']][record['sourceHash']] = float(record['studentLogit']) - STUDENT_OFFSET

    mirror_run = _read_jsonl(ART / 'mcp-benchmarks/0b50fc0a7b744a42b0e7b0e1f11df5a4/scores.jsonl')
    student_run = _read_jsonl(ART / 'mcp-benchmarks/583f8f603f9f4109a82703756a099936/scores.jsonl')
    mirror_reddit = {}
    reconstructed_reddit = {}
    student_reddit = {}
    for record in mirror_run:
        if record['sourceHash'] not in reddit_hashes:
            continue
        mirror_reddit[record['sourceHash']] = float(record['components']['mirror']['logit'])
        reconstructed_reddit[record['sourceHash']] = community_ferret_logit(
            float(record['components']['community_forensics']['logit']),
            float(record['components']['ferretnet_b']['logit']),
        )
    for record in student_run:
        if record['sourceHash'] not in reddit_hashes:
            continue
        student_reddit[record['sourceHash']] = float(record['logit'])

    checks = _verify(calibration_hashes, reddit_hashes, ferret_cal, mirror_reddit, reconstructed_reddit, student_reddit)
    if checks['communityFerretReddit']['maxAbsDifference'] is None or checks['communityFerretReddit']['maxAbsDifference'] > 1e-4:
        reconstructed_reddit = {}
    return {
        'sources': {
            'mirrorCalibration': 'artifacts/mirror-evaluation-2026-10-13/comparison-scores.csv system=MIRROR split=tuning',
            'communityFerretCalibration': 'artifacts/mirror-evaluation-2026-10-13/comparison-scores.csv system=Community + Ferret split=tuning',
            'studentCalibration': 'artifacts/reddit-adaptation-2026-10-14/student-calibration-transfer.json studentLogit minus packaged logitOffset',
            'mirrorReddit': 'artifacts/mcp-benchmarks/0b50fc0a7b744a42b0e7b0e1f11df5a4 components.mirror.logit',
            'communityFerretReddit': 'reconstructed max(community, ferret - logit(0.8304235935211183)) from the same run, kept only after the check below',
            'studentReddit': 'artifacts/mcp-benchmarks/583f8f603f9f4109a82703756a099936 top-level decision logit',
        },
        'studentOffset': STUDENT_OFFSET,
        'ferretThreshold': FERRET_THRESHOLD,
        'ferretOffset': FERRET_OFFSET,
        'checks': checks,
        'logits': {
            'calibration-tuning': {
                'original': _pack(calibration, 'original', mirror_cal['original'], ferret_cal['original'], student_cal['original']),
                'reddit-webp640': _pack(calibration, 'reddit-webp640', mirror_cal['reddit-webp640'], ferret_cal['reddit-webp640'], student_cal['reddit-webp640']),
            },
            'reddit-preview-tuning': {
                'preview': _pack(reddit, 'preview', mirror_reddit, reconstructed_reddit, student_reddit),
            },
        },
    }


def _pack(rows: list[dict], variant: str, mirror: dict, ferret: dict, student: dict) -> dict:
    packed = {'MIRROR': {}, 'Community+Ferret': {}, 'reddit-student-last2-v1': {}}
    for row in rows:
        digest = row['sourceHash']
        if digest in mirror:
            packed['MIRROR'][digest] = mirror[digest]
        if digest in ferret:
            packed['Community+Ferret'][digest] = ferret[digest]
        if digest in student:
            packed['reddit-student-last2-v1'][digest] = student[digest]
    packed['coverage'] = {name: len(values) for name, values in packed.items() if name != 'coverage'}
    packed['variant'] = variant
    packed['rows'] = len(rows)
    return packed


def _verify(calibration, reddit, ferret_csv, mirror_reddit, reconstructed, student_reddit) -> dict:
    original = _read_jsonl(ART / 'mcp-benchmarks/269057a61047438899dec5e50fb8c7d0/scores.jsonl')
    webp = _read_jsonl(ART / 'mcp-benchmarks/c7dcabd3d9b445c2b3bcc80c51dfbc19/scores.jsonl')
    pairs = []
    for record in original + webp:
        if record['sourceHash'] not in calibration:
            continue
        combined = community_ferret_logit(
            float(record['components']['community_forensics']['logit']),
            float(record['components']['ferretnet_b']['logit']),
        )
        pairs.append((combined, float(record['logit'])))
    csv_pairs = []
    for record in original:
        digest = record['sourceHash']
        if digest in ferret_csv['original']:
            csv_pairs.append((float(record['logit']), ferret_csv['original'][digest]))
    reddit_csv = {}
    with (ART / 'mirror-evaluation-2026-10-13/comparison-scores.csv').open() as handle:
        for record in csv.DictReader(handle):
            if record['dataset'] == 'reddit-mirror' and record['split'] == 'tuning' and record['system'] == 'Community + Ferret':
                if record['sourceHash'] in reddit:
                    reddit_csv[record['sourceHash']] = float(record['logit'])
    reddit_pairs = [(reconstructed[digest], reddit_csv[digest]) for digest in reconstructed if digest in reddit_csv]
    mirror_csv = {}
    with (ART / 'mirror-evaluation-2026-10-13/comparison-scores.csv').open() as handle:
        for record in csv.DictReader(handle):
            if record['dataset'] == 'reddit-mirror' and record['split'] == 'tuning' and record['system'] == 'MIRROR':
                if record['sourceHash'] in reddit:
                    mirror_csv[record['sourceHash']] = float(record['logit'])
    mirror_pairs = [(mirror_reddit[digest], mirror_csv[digest]) for digest in mirror_reddit if digest in mirror_csv]
    tuning_predictions = json.loads((ART / 'reddit-adaptation-2026-10-14/student-tuning-predictions.json').read_text())
    student_pairs = []
    for record in tuning_predictions['rows']:
        if record.get('eligible') and record['sourceHash'] in student_reddit:
            student_pairs.append((float(record['logit']) - STUDENT_OFFSET, student_reddit[record['sourceHash']]))
    return {
        'communityFerretFormulaVsStoredCombined': {'n': len(pairs), 'maxAbsDifference': _max_abs(pairs)},
        'communityFerretStoredVsComparisonCsv': {'n': len(csv_pairs), 'maxAbsDifference': _max_abs(csv_pairs)},
        'communityFerretReddit': {'n': len(reddit_pairs), 'overlapWithComparisonCsv': len(reddit_csv), 'maxAbsDifference': _max_abs(reddit_pairs)},
        'mirrorRedditVsComparisonCsv': {'n': len(mirror_pairs), 'maxAbsDifference': _max_abs(mirror_pairs)},
        'studentRedditVsTuningPredictions': {'n': len(student_pairs), 'maxAbsDifference': _max_abs(student_pairs)},
    }


def aligned(rows: list[dict], mapping: dict[str, float]) -> tuple[list[dict], np.ndarray]:
    kept = [row for row in rows if row['sourceHash'] in mapping]
    logits = np.array([mapping[row['sourceHash']] for row in kept], dtype=float)
    return kept, logits


if __name__ == '__main__':
    calibration = calibration_rows()
    reddit, summary = reddit_rows()
    payload = load_baselines(calibration, reddit)
    print(json.dumps({'reddit': summary, 'checks': payload['checks'], 'coverage': {
        key: {variant: block['coverage'] for variant, block in value.items()}
        for key, value in payload['logits'].items()
    }}, indent=2))
