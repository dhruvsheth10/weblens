import {readFile, writeFile} from 'node:fs/promises';
import {calibrate, report, validateRows} from '../src/detection/metrics';
const args = process.argv.slice(2);
const input = args[0];
const thresholdArg = args.indexOf('--threshold');
const outputArg = args.indexOf('--out');
try {
  if (!input) throw new Error('Usage: npm run evaluate -- scores.json [--threshold 0.8] [--out report.json]');
  const data = JSON.parse(await readFile(input, 'utf8'));
  const rows = validateRows(data.rows);
  if (data.failures?.length) throw new Error('The run contains failed/skipped images. Resolve failures and rerun the collection before evaluation.');
  const calibration = rows.filter(row => row.split === 'calibration');
  const validation = rows.filter(row => row.split === 'validation');
  if (!validation.some(r => r.label === 'real') || !validation.some(r => r.label === 'ai')) throw new Error('Held-out validation requires both real and AI images.');
  const threshold = thresholdArg >= 0 ? Number(args[thresholdArg + 1]) : calibrate(calibration);
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1 + Number.EPSILON) throw new Error('Invalid threshold.');
  const validationReport = report(validation, threshold);
  const original = validation.filter(r => r.variant === 'original');
  const realCount = new Set(validation.filter(r => r.label === 'real').map(r => r.sourceHash)).size;
  const warnings = [
    'Raw sigmoid scores are not calibrated probabilities. The threshold is not applied to the extension automatically.',
    'Representative independent data and accurate labels are required; this tool cannot verify those properties.',
    'Derived transformations are correlated. Use unique source images for sample-size claims.',
  ];
  if (realCount < 300) warnings.push(`Only ${realCount} unique real validation images. Even zero false flags is insufficient evidence for a 1% upper bound at 95% confidence.`);
  if (threshold > 1) warnings.push('No deployable threshold satisfies calibration constraints: flags must be disabled or the detector improved.');
  const result = {
    model: rows[0].model, modelHash: rows[0].modelHash, pipeline: rows[0].pipeline, engine: rows[0].engine, targetFpr: 0.01,
    thresholdSource: thresholdArg >= 0 ? 'fixed in advance by user' : 'calibration split, constrained by category and transformation',
    threshold, calibration: calibration.length ? report(calibration, threshold) : null,
    validation: validationReport, originalValidation: report(original, threshold),
    empiricalFprTargetMet: threshold <= 1 && validationReport.fpr! <= 0.01 && Object.values(validationReport.groups).every(group => group.fpr === null || group.fpr <= 0.01),
    warnings,
  };
  const text = JSON.stringify(result, null, 2);
  if (outputArg >= 0) {
    if (!args[outputArg + 1]) throw new Error('--out requires a path.');
    await writeFile(args[outputArg + 1], text + '\n');
  }
  console.log(text);
  if (!result.empiricalFprTargetMet) process.exitCode = 2;
} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
