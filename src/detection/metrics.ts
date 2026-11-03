import {MODELS, sigmoid, type BenchmarkRow, type ModelId} from './model';
export function metrics(rows: Pick<BenchmarkRow, 'score' | 'label'>[], threshold: number) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const row of rows) {
    if (!Number.isFinite(row.score) || row.score < 0 || row.score > 1) throw new Error('Scores must be finite and between 0 and 1.');
    if (row.label !== 'real' && row.label !== 'ai') throw new Error('Unknown ground-truth label.');
    if (row.label === 'ai') row.score >= threshold ? tp++ : fn++;
    else row.score >= threshold ? fp++ : tn++;
  }
  return {tp, fp, tn, fn, fpr: fp + tn ? fp / (fp + tn) : null, recall: tp + fn ? tp / (tp + fn) : null, precision: tp + fp ? tp / (tp + fp) : null};
}
export function recommendThreshold(rows: Pick<BenchmarkRow, 'score' | 'label'>[], maxFpr = 0.01) {
  if (maxFpr < 0 || maxFpr >= 1) throw new Error('Invalid false-positive target.');
  if (!rows.some(r => r.label === 'real') || !rows.some(r => r.label === 'ai')) throw new Error('Calibration needs both real and AI images.');
  metrics(rows, 0.5); // Validate before sorting.
  const reals = rows.filter(r => r.label === 'real').map(r => r.score).sort((a, b) => b - a);
  const allowed = Math.floor(maxFpr * reals.length);
  // Strictly exceed the next forbidden real score, including ties and saturated 1.0 scores.
  return reals[allowed] + Number.EPSILON;
}
export function validateRows(value: unknown): BenchmarkRow[] {
  if (!Array.isArray(value) || !value.length) throw new Error('No benchmark rows found.');
  const first = value[0];
  if (!first || typeof first !== 'object' || !(first.model in MODELS)) throw new Error('Unknown detector version.');
  const expected = MODELS[first.model as ModelId];
  const seen = new Map<string, BenchmarkRow>();
  const variants = new Set<string>();
  for (const row of value) {
    if (!row || typeof row !== 'object' || row.model !== expected.id || row.modelHash !== expected.sha256) throw new Error('Model versions do not match. Benchmark each detector separately.');
    if (row.pipeline !== expected.pipeline) throw new Error('Preprocessing versions do not match the selected detector. Rerun this collection.');
    if (!['cpu', 'mps', 'cuda'].includes(row.engine) || row.engine !== value[0].engine) throw new Error('Benchmark CPU, MPS and CUDA separately; their numerical results can differ.');
    if (!['calibration', 'validation'].includes(row.split) || !['original', 'jpeg75', 'resize512'].includes(row.variant)) throw new Error('Invalid split or transformation.');
    if (typeof row.file !== 'string' || !row.file || typeof row.group !== 'string' || !row.group || !/^[a-f0-9]{64}$/.test(row.sourceHash)) throw new Error('A file, category and source SHA-256 are required for every row.');
    if (!Number.isFinite(row.logit) || Math.abs(sigmoid(row.logit) - row.score) > 1e-7) throw new Error('The raw score does not match its logit.');
    const previous = seen.get(row.sourceHash);
    if (previous && (previous.split !== row.split || previous.label !== row.label || previous.file !== row.file || previous.group !== row.group)) throw new Error('Duplicate source image or leakage between calibration and validation.');
    const key = `${row.sourceHash}:${row.variant}`;
    if (variants.has(key)) throw new Error('Duplicate image/transformation row.');
    variants.add(key); seen.set(row.sourceHash, row);
  }
  metrics(value, 0.5);
  return value;
}
export function calibrate(rows: BenchmarkRow[], maxFpr = 0.01) {
  if (rows.some(row => row.split !== 'calibration')) throw new Error('Only calibration rows can choose the threshold.');
  const all = recommendThreshold(rows, maxFpr);
  const strata = [...new Set(rows.filter(r => r.label === 'real').map(r => `${r.group}/${r.variant}`))];
  const thresholds = strata.map(stratum => {
    const realRows = rows.filter(r => r.label === 'real' && `${r.group}/${r.variant}` === stratum);
    // Add an AI row only to satisfy the public helper's both-label contract; it cannot set this threshold.
    return recommendThreshold([...realRows, rows.find(r => r.label === 'ai')!], maxFpr);
  });
  return Math.max(all, ...thresholds);
}
export function report(rows: BenchmarkRow[], threshold: number) {
  const groups = [...new Set(rows.map(r => `${r.group} / ${r.variant}`))];
  return {
    threshold, total: rows.length, uniqueImages: new Set(rows.map(r => r.sourceHash)).size, ...metrics(rows, threshold),
    groups: Object.fromEntries(groups.map(group => [group, metrics(rows.filter(r => `${r.group} / ${r.variant}` === group), threshold)])),
    // One-sided exact 95% upper bound is only provided when no false positives occur.
    zeroFpUpperBound95: rows.some(r => r.label === 'real') && metrics(rows, threshold).fp === 0
      ? 1 - Math.pow(0.05, 1 / new Set(rows.filter(r => r.label === 'real').map(r => r.sourceHash)).size) : null,
  };
}
