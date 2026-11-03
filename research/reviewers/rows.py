"""Authorized evaluation rows only.

Calibration: experimentSplit == tuning under Downloads/benchmark/calibration.
Reddit: image-map preview rows with usageSplit == tuning, dropping label-audit
rows whose eligibleForWeakSupervision is false. Holdout, quarantine, audit,
and development rows are not opened.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_ROOT = Path('/Users/dhruvsheth/Downloads/benchmark/calibration')
VALIDATION_ROOT = Path('/Users/dhruvsheth/Downloads/benchmark/validation')
REDDIT_ROOT = ROOT / 'datasets/reddit/data'
MANIFEST = ROOT / 'artifacts/ferretnet-improvement-2026-10-11/manifest.json'
IMAGE_MAP = ROOT / 'artifacts/reddit-adaptation-2026-10-14/image-map.json'
LABEL_AUDIT = ROOT / 'artifacts/reddit-adaptation-2026-10-14/label-audit.json'


def assert_allowed(path: Path) -> None:
    resolved = path.resolve()
    if resolved == VALIDATION_ROOT or resolved.is_relative_to(VALIDATION_ROOT):
        raise RuntimeError(f'Refusing validation path {path}')


def calibration_rows() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for record in manifest['rows']:
        if record.get('experimentSplit') != 'tuning':
            continue
        relative = Path(record['file'])
        if relative.parts[0] != 'calibration':
            raise ValueError(f'Unexpected calibration file {record["file"]}')
        path = CALIBRATION_ROOT / Path(*relative.parts[1:])
        assert_allowed(path)
        rows.append({
            'set': 'calibration-tuning',
            'id': record.get('sourceId') or record['file'],
            'file': record['file'],
            'path': str(path),
            'root': str(CALIBRATION_ROOT),
            'sourceHash': record['sourceHash'],
            'label': record['label'],
            'category': record['group'],
            'variants': ['original', 'reddit-webp640'],
        })
    return rows


def reddit_rows() -> tuple[list[dict], dict]:
    image_map = json.loads(IMAGE_MAP.read_text())
    audit = {row['id']: row for row in json.loads(LABEL_AUDIT.read_text())['rows']}
    kept = []
    skipped = []
    for record in image_map['rows']:
        if record.get('role') != 'preview' or record.get('usageSplit') != 'tuning':
            continue
        entry = audit.get(record['id'])
        if entry is None:
            raise ValueError(f'Preview tuning id {record["id"]} has no label-audit entry')
        if entry.get('eligibleForWeakSupervision') is False:
            skipped.append({'id': record['id'], 'reviewReasons': entry.get('reviewReasons') or []})
            continue
        if entry.get('eligibleForWeakSupervision') is not True:
            raise ValueError(f'Unexpected eligibility for {record["id"]}')
        path = REDDIT_ROOT / record['preview']['path']
        assert_allowed(path)
        kept.append({
            'set': 'reddit-preview-tuning',
            'id': record['id'],
            'file': record['file'],
            'path': str(path),
            'root': str(REDDIT_ROOT),
            'sourceHash': record['sourceHash'],
            'label': record['label'],
            'category': record['category'],
            'variants': ['preview'],
        })
    summary = {
        'previewTuning': len(kept) + len(skipped),
        'skippedIneligible': len(skipped),
        'scored': len(kept),
        'skipReasons': skipped,
    }
    return kept, summary
