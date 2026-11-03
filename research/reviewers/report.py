"""Render artifacts/reviewer-screen-2026-10-16/report.md from scored predictions."""
from __future__ import annotations

import json
import re
from pathlib import Path


def _fmt(value, digits=3):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _point(block: dict, cap: str) -> dict:
    return block['optimisticInSample'][cap]


def _line(name: str, block: dict | None, latency: dict | None = None) -> str:
    if not block:
        return f'| {name} | — | — | — | — | — | — |'
    point1 = _point(block, '0.01')
    point5 = _point(block, '0.05')
    median = _fmt(latency['medianMs'], 1) if latency else '—'
    return (
        f"| {name} | {block['n']} | {_fmt(block['auroc'])} | {_fmt(block['partialAuRoc01'])} | "
        f"{_fmt(point1['tpr'])} | {_fmt(point5['tpr'])} | {median} |"
    )


def _category_table(block: dict) -> str:
    lines = ['| Category | Real n | FPR at in-sample 1% threshold |', '|---|---:|---:|']
    categories = _point(block, '0.01')['perCategory']
    for name, value in categories.items():
        fpr = '—' if value['fpr'] is None else _fmt(value['fpr'])
        lines.append(f"| {name} | {value['realN']} | {fpr} |")
    return '\n'.join(lines)


def _verdict(metadata: dict) -> str:
    models = metadata.get('models') or {}
    sentences = []
    for name in ('dda', 'omniaid'):
        entry = models.get(name)
        if not entry or not entry.get('ran'):
            reason = (entry or {}).get('error', 'it was not scored')
            sentences.append(f'{name} did not produce a usable screen ({reason[:400]}).')
            continue
        metrics = entry.get('metrics') or {}
        reddit = metrics.get('reddit-preview-tuning/preview')
        if not reddit or not reddit['baselines'].get('MIRROR', {}).get('available'):
            sentences.append(f'{name} ran, but MIRROR logits were not available on the same Reddit rows, so it is not ranked against MIRROR.')
            continue
        mirror = reddit['baselines']['MIRROR']
        candidate = mirror['candidateOnIntersection']
        baseline = mirror['baseline']
        candidate_tpr = _point(candidate, '0.01')['tpr']
        mirror_tpr = _point(baseline, '0.01')['tpr']
        better_rank = candidate['auroc'] > baseline['auroc'] + 0.01 and candidate_tpr > mirror_tpr + 0.02
        if better_rank:
            sentences.append(
                f"{name} ranks Reddit tuning previews better than MIRROR on the shared rows "
                f"(AUROC {_fmt(candidate['auroc'])} vs {_fmt(baseline['auroc'])}, "
                f"in-sample TPR at 1% FPR {_fmt(candidate_tpr)} vs {_fmt(mirror_tpr)}). "
                'That is evidence it could help triage, still on weak source labels and an in-sample threshold.'
            )
        else:
            sentences.append(
                f"{name} is not a better reviewer than MIRROR on the old Reddit tuning previews. "
                f"On the shared rows its AUROC is {_fmt(candidate['auroc'])} versus MIRROR {_fmt(baseline['auroc'])}, "
                f"and its in-sample TPR at 1% FPR is {_fmt(candidate_tpr)} versus {_fmt(mirror_tpr)}."
            )
        webp = metrics.get('calibration-tuning/reddit-webp640')
        if webp and webp.get('candidate', {}).get('auroc') is not None:
            webp_auroc = webp['candidate']['auroc']
            mirror_webp = None
            mirror_block = (webp.get('baselines') or {}).get('MIRROR') or {}
            if mirror_block.get('available'):
                mirror_webp = mirror_block['baseline']['auroc']
            if webp_auroc < 0.5:
                extra = ''
                if name == 'dda':
                    extra = (
                        ' After the width-640 cap and before the 336 center crop, only 24 of those 483 images '
                        'have a short side under 336 (22 real, 2 AI), so zero-padding from CenterCrop does not explain the inversion.'
                    )
                sentences.append(
                    f"On the calibration tuning WebP640 variant, {name} AUROC is {_fmt(webp_auroc)}, "
                    'below 0.5, with the same official preprocessing used on the original files.' + extra
                )
            elif mirror_webp is not None and webp_auroc > mirror_webp + 0.01:
                sentences.append(
                    f"On calibration tuning WebP640, {name} AUROC {_fmt(webp_auroc)} is above MIRROR {_fmt(mirror_webp)}. "
                    'Community + Ferret and the Reddit student still rank higher on that variant. '
                    'The Reddit preview rows are the set that matches the reviewer job.'
                )
    if not sentences:
        return 'Neither candidate was scored.'
    return ' '.join(sentences)


def write_report(out: Path, metadata: dict, reddit_summary: dict) -> None:
    row_summary_path = out / 'row-summary.json'
    row_summary = json.loads(row_summary_path.read_text()) if row_summary_path.exists() else {}
    baselines_path = out / 'baselines.json'
    baselines = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}
    checks = baselines.get('checks') or {}
    lines = [
        '# Reviewer screen — 2026-10-16',
        '',
        'DDA and OmniAID were scored as offline reviewer candidates. They are not part of the extension. '
        'Only calibration rows with `experimentSplit == tuning` and old Reddit preview rows with `role == preview` '
        'and `usageSplit == tuning` were opened. Audit, development, holdout, quarantine, and the validation directory were not scored.',
        '',
        '## Setup',
        '',
        f"- Torch {metadata.get('torch')}. MPS available at launch: {metadata.get('mpsAvailable')}.",
        f"- Third-party clones: DDA `{((metadata.get('revisions') or {}).get('dda'))}`, OmniAID `{((metadata.get('revisions') or {}).get('omniaid'))}`.",
        '- No files under the cloned repos were edited. Device is chosen by this runner (`mps`, then `cpu` if a forward fails). Their demo scripts still default to CUDA and were not executed.',
        '- Third-party `requirements.txt` for OmniAID pins CUDA wheels (`--extra-index-url https://download.pytorch.org/whl/cu126`, torch 2.8). Those were not installed. A separate venv at `research/reviewers/.venv` was created without `--system-site-packages` (that flag does not see a parent venv). `research/reviewers/.venv/lib/python3.12/site-packages/backend-venv.pth` points at `backend/.venv` site-packages so the existing Mac torch is reused. `backend/.venv` was not modified.',
        '- WebP640 images are produced by importing `backend.image_io.decode_image(data, "reddit-webp640")`, which is the Lanczos width-capped WebP quality-80 simulation in `backend/previews.py`. Reddit preview rows are decoded with `decode_image(data, "original")` and are not resized again.',
        '- Every image is checked with `research.reddit_adaptation.common.safe_data` (sha256 equals `sourceHash`, path stays inside the declared root) before it is decoded.',
        f"- Calibration tuning rows: {row_summary.get('calibrationTuning')}. Reddit preview tuning rows: {reddit_summary.get('previewTuning')}, skipped because `eligibleForWeakSupervision` is false: {reddit_summary.get('skippedIneligible')}, scored: {reddit_summary.get('scored')}. All 18 skips are hard-case gallery panels whose post-level provenance does not establish that panel.",
        '- Thresholds for TPR at 1% and 5% FPR are chosen on the same rows they are measured on. They are optimistic, not deployment thresholds. Partial AUROC is `summarize`\'s sklearn `roc_auc_score(..., max_fpr=0.01)`, which is the McClish-standardized partial AUC, not the raw area from 0 to 0.01.',
        '- The packaged Reddit student decision logit is the stored raw logit minus `logitOffset` from `student-profile.json`. A constant shift does not change AUROC or TPR at a fixed FPR. Community + Ferret on calibration is the stored combined logit. On Reddit previews it is `max(community_logit, ferret_logit - logit(0.8304235935211183))` from MCP run `0b50fc0a` component logits. That formula matches stored combined logits with max absolute difference 0 on 966 calibration rows and on the 264 Reddit tuning rows that also appear in the comparison CSV. The other eligible previews have no stored combined logit, so their Community + Ferret score is the same formula without a further per-row check.',
        '',
        '## Licenses and checkpoints',
        '',
        '- DDA code and `DDA_ckpt.pth` are Apache-2.0 (repository LICENSE and the Hugging Face card). Backbone is DINOv2 ViT-L/14 via `torch.hub` (`facebookresearch/dinov2`), which carries the DINOv2 license.',
        '- OmniAID weights are MIT on the Hugging Face card. The GitHub repository has no LICENSE file, so the code\'s license is not stated there. The runner uses the reward modules from that repository.',
        '',
        '### Preprocessing',
        '',
    ]
    for name, entry in (metadata.get('models') or {}).items():
        info = (entry or {}).get('info') or {}
        lines.append(f"- **{name}** ran={entry.get('ran')}. {info.get('preprocessing', 'No preprocessing recorded.')}")
        if info.get('loadNote'):
            lines.append(f"  - {info['loadNote']}")
        lines.append(
            f"  - weights_only={info.get('weightsOnly')} device={info.get('device')} "
            f"missing_keys={info.get('missing')} unexpected_keys={info.get('unexpected')} "
            f"checkpoint=`{info.get('checkpoint')}`"
        )
        if info.get('weightsOnlyError'):
            cleaned = re.sub(r'\x1b\[[0-9;]*m', '', str(info['weightsOnlyError']))
            match = re.search(r'Unsupported global: GLOBAL \S+', cleaned)
            detail = match.group(0) if match else cleaned.split('\n', 1)[0]
            lines.append(
                f"  - weights_only=True failed ({detail}). "
                'The official checkpoint was then loaded with weights_only=False.'
            )
        if info.get('mpsError'):
            lines.append(f"  - MPS error: {info['mpsError']}")
        if info.get('patches'):
            lines.append(f"  - Patches: {info['patches']}")
    lines.extend(['', '### Baseline reconstruction checks', ''])
    for name, check in checks.items():
        lines.append(f"- {name}: n={check.get('n')} max absolute logit difference={check.get('maxAbsDifference')}")
    lines.extend([
        '',
        '## Results',
        '',
        'Partial AUROC is the standardized sklearn value from `summarize`. TPR columns use an optimistic in-sample threshold. '
        'Latency is milliseconds per image for preprocessing plus forward, including the first warmup call in the mean. Median is the better typical cost. '
        'Baseline rows are the intersection with rows that candidate scored. When the intersection is the full candidate set, the candidate row and the "candidate on intersection" row match.',
        '',
    ])
    for name, entry in (metadata.get('models') or {}).items():
        lines.append(f'### {name}')
        lines.append('')
        if not entry.get('ran'):
            lines.append(f"Did not run. {entry.get('error', '')[:1500]}")
            lines.append('')
            continue
        latency = None
        for split, block in (entry.get('metrics') or {}).items():
            lines.append(f'#### {split}')
            lines.append('')
            lines.append('| Model | n | AUROC | partial AUROC @ 1% FPR | TPR @ FPR≤1% | TPR @ FPR≤5% | median ms |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|')
            lines.append(_line(name, block['candidate'], block.get('latency')))
            missing = []
            for baseline_name, baseline in block['baselines'].items():
                if not baseline.get('available'):
                    missing.append(baseline_name)
                    continue
                if baseline['intersection'] != block['candidate']['n']:
                    lines.append(_line(f'{name} on {baseline_name} intersection', baseline['candidateOnIntersection']))
                lines.append(_line(baseline_name, baseline['baseline']))
            lines.append('')
            if missing:
                lines.append('No stored logits for: ' + ', '.join(missing) + '.')
                lines.append('')
            lines.append(f'Per-category false-positive rate for {name} at its own in-sample 1% FPR threshold:')
            lines.append('')
            lines.append(_category_table(block['candidate']))
            lines.append('')
            agree = block.get('agreement') or {}
            if agree:
                lines.append(
                    f"Disagreements with MIRROR at each model's own in-sample 1% FPR threshold: "
                    f"{agree.get('disagreements')} of {agree.get('n')}. "
                    f"{name} was right on {agree.get('candidateRight')}; MIRROR was right on {agree.get('mirrorRight')}."
                )
                lines.append('')
            latency = block.get('latency')
        if latency:
            lines.append(
                f"Latency on the last split above: mean {_fmt(latency['meanMs'], 1)} ms, "
                f"median {_fmt(latency['medianMs'], 1)} ms, p90 {_fmt(latency['p90Ms'], 1)} ms, "
                f"first call {_fmt(latency['firstMs'], 1)} ms, n={latency['n']}."
            )
            lines.append('')
    disk = metadata.get('diskBytes') or {}
    lines.extend(['## Disk', ''])
    if not disk:
        lines.append('Disk usage was not recorded.')
    for path, size in disk.items():
        if size >= 1024**3:
            shown = f'{size / 1024**3:.2f} GiB'
        else:
            shown = f'{size / 1024**2:.1f} MiB'
        lines.append(f'- `{path}`: {shown}')
    third_party = next((size for path, size in disk.items() if path.endswith('third_party')), 0)
    if third_party > 8 * 1024**3:
        lines.append(
            '- That third_party tree is above the 8 GiB download budget. It includes the git clones, '
            '`DDA_ckpt.pth` (1.17 GiB), `checkpoint_omniaid_v2.pth` (3.02 GiB), the DINOv2 ViT-L/14 hub weights (about 1.1 GiB), '
            'and the CLIP ViT-L/14@336 cache under `hf_home` (about 3.2 GiB). The OmniAID DINOv3 checkpoint was not downloaded.'
        )
    lines.extend(['', '## Verdict', '', _verdict(metadata), ''])
    lines.append(
        'These labels are source-derived. Wilson intervals in the JSON summaries treat images as independent and do not capture source-label errors or correlated posts. '
        'In-sample thresholds overstate the true positive rate that a frozen threshold would have on new posts.'
    )
    lines.append('')
    (out / 'report.md').write_text('\n'.join(lines))
