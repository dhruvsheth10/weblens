import test from 'node:test';
import assert from 'node:assert/strict';
import {MODEL, PIPELINE, sigmoid, isRedditImageUrl, decisionBand, verdict, type BenchmarkRow} from '../src/detection/model';
import {metrics, recommendThreshold, calibrate, validateRows, report} from '../src/detection/metrics';
test('raw sigmoid is stable at extreme logits and rejects corrupt output', () => {
  assert.equal(sigmoid(0), 0.5); assert.equal(sigmoid(1000), 1); assert.equal(sigmoid(-1000), 0);
  assert.throws(() => sigmoid(NaN)); assert.throws(() => sigmoid(Infinity));
});
test('semantic decisions preserve high, uncertain and unmarked boundaries',()=>{
  assert.equal(decisionBand(.5,.5,.04),'ai');
  assert.equal(decisionBand(.04,.5,.04),'uncertain');
  assert.equal(decisionBand(.039,.5,.04),'unmarked');
  assert.equal(verdict(.2,.5,.04),'Uncertain AI signal');
  assert.equal(decisionBand(.55,.6,.04),'uncertain');
  assert.equal(decisionBand(.02,.02,.04),'ai');
  assert.equal(verdict(.2,.5),'No strong AI signal');
  assert.equal(decisionBand(.4,.5,.04,.01),'unmarked');
  assert.equal(decisionBand(.5,.5,.04,.01),'ai');
});
test('image requests cannot target arbitrary URLs, credentials or host lookalikes', () => {
  assert.equal(isRedditImageUrl('https://preview.redd.it/photo.jpg?width=640'), true);
  for (const url of ['http://i.redd.it/a.png', 'https://i.redd.it.evil.test/a.png', 'https://user@i.redd.it/a.png', 'https://i.redd.it:8443/a.png', 'file:///tmp/a.png', 'https://127.0.0.1/a.png']) assert.equal(isRedditImageUrl(url), false);
});
test('threshold selection handles tied real scores and impossible saturated scores', () => {
  const rows = [{label: 'real' as const, score: .9}, {label: 'real' as const, score: .9}, {label: 'ai' as const, score: .95}];
  const threshold = recommendThreshold(rows);
  assert.ok(threshold > .9); assert.equal(metrics(rows, threshold).fp, 0); assert.equal(metrics(rows, threshold).tp, 1);
  assert.ok(recommendThreshold([{label: 'real', score: 1}, {label: 'ai', score: 1}]) > 1);
  assert.throws(() => recommendThreshold([{label: 'ai', score: .9}]));
});
function row(index: number, label: 'real' | 'ai', score: number, patch: Partial<BenchmarkRow> = {}): BenchmarkRow {
  return {model: MODEL.id, modelHash: MODEL.sha256, pipeline: PIPELINE, file: `${index}.png`, sourceHash: index.toString(16).padStart(64, '0'), label, score, logit: Math.log(score / (1 - score)), width: 640, height: 480, elapsedMs: 30, engine: 'mps', variant: 'original', split: 'calibration', group: 'photo', ...patch};
}
test('calibration protects rare difficult categories instead of hiding them in overall accuracy', () => {
  const rows = Array.from({length: 200}, (_, i) => row(i + 1, 'real', .1));
  rows.push(row(201, 'real', .98, {group: 'artwork'}), row(202, 'ai', .99));
  assert.ok(calibrate(rows) > .98);
  assert.throws(() => calibrate([row(1, 'real', .2, {split: 'validation'}), row(2, 'ai', .9)]));
});
test('benchmark input rejects split leakage, duplicated files and mixed detector versions', () => {
  const real = row(1, 'real', .1), ai = row(2, 'ai', .9);
  assert.equal(validateRows([real, ai]).length, 2);
  assert.throws(() => validateRows([real, {...real, split: 'validation'}]));
  assert.throws(() => validateRows([real, {...real, file: 'duplicate.png', variant: 'jpeg75'}]));
  assert.throws(() => validateRows([real, {...ai, modelHash: 'old-model'}]));
  assert.throws(() => validateRows([real, {...ai, pipeline: 'old-transform'}]));
  assert.throws(() => validateRows([real, {...ai, engine: 'cpu'}]));
  assert.throws(() => validateRows([real, {...ai, score: .7}]));
});
test('uncertainty counts unique images, not repeat transformations', () => {
  const real = row(1, 'real', .1, {split: 'validation'});
  const result = report([real, {...real, variant: 'jpeg75'}, {...real, variant: 'resize512'}], .99);
  assert.equal(result.uniqueImages, 1); assert.ok(Math.abs(result.zeroFpUpperBound95! - .95) < 1e-9);
  assert.equal(metrics([row(1, 'real', .99), row(2, 'ai', .2)], .99).fp, 1);
});
