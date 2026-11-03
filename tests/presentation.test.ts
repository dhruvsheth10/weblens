import test from 'node:test';
import assert from 'node:assert/strict';
import {detectedMark,summarizeResults} from '../src/extension/presentation';
import {MODELS,type Detection} from '../src/detection/model';
const MODEL=MODELS['community-ferret-preview-v1'];
function detection(score:number,semanticScore=score):Detection{return {model:MODEL.id,modelHash:MODEL.sha256,pipeline:MODEL.pipeline,score,semanticScore,logit:Math.log(score/(1-score)),variant:'original',engine:'cpu',width:256,height:256,elapsedMs:1};}
test('low results visibly confirm scanning without asserting real origin',()=>{
  const mark=detectedMark(detection(.001),MODEL,.5);
  assert.equal(mark.state,'unmarked');assert.match(mark.label,/Scanned/);assert.match(mark.title,/does not establish that an image is real/);
});
test('display rounding never changes the actual flag boundary or semantic uncertainty rule',()=>{
  assert.equal(detectedMark(detection(.499999),MODEL,.5).state,'uncertain');
  assert.equal(detectedMark(detection(.5),MODEL,.5).state,'ai');
  assert.equal(detectedMark(detection(.4,.001),MODEL,.5).state,'unmarked');
});
test('preview and recording-failure marks distinguish incomplete or unsaved results',()=>{
  assert.equal(detectedMark(detection(.8),MODEL,.5,true).busy,true);
  const mark=detectedMark({...detection(.8),recordingError:'Disk unavailable'},MODEL,.5);
  assert.equal(mark.state,'ai');assert.match(mark.detail,/recording failed/);
});
test('progress and failures do not count as successfully scanned negative images',()=>{
  const results=new Map([['same-url',{state:'ai' as const}],['uncertain',{state:'uncertain' as const}],['low',{state:'unmarked' as const}],['failed',{state:'error' as const}],['in-progress',{state:'scanning' as const}]]);
  results.set('same-url',{state:'ai'});
  assert.deepEqual(summarizeResults(results.values()),{scanned:3,ai:1,uncertain:1,errors:1});
});
