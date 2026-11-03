import test from 'node:test';
import assert from 'node:assert/strict';
import {VALIDATION_NOTE, VALIDATION_ROWS} from '../src/detection/validationBenchmark';

test('published validation table keeps the packaged 17 Oct 2026 comparison', () => {
  assert.equal(VALIDATION_ROWS.length, 54);
  const models = new Set(VALIDATION_ROWS.map(row => row.model));
  assert.equal(models.size, 9);
  assert.equal(models.has('fiser-dear-r-v1'), false);
  const sherlock = VALIDATION_ROWS.find(row => row.model === 'reddit-sherlock-2' && row.variant === 'original' && row.scope === 'full-folder');
  assert.ok(sherlock);
  assert.equal(sherlock.cutoff, 0.5);
  assert.equal(sherlock.tp, 747);
  assert.equal(sherlock.fp, 1);
  assert.equal(sherlock.fn, 502);
  assert.equal(VALIDATION_NOTE.newUniqueAi, 24);
  assert.equal(VALIDATION_NOTE.calibrationOverlap, 1227);
});
