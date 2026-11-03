import test from 'node:test';
import assert from 'node:assert/strict';
import {backendCompatibility} from '../src/extension/health';
import {sameImage} from '../src/extension/lifecycle';
import {MODELS} from '../src/detection/model';

test('backend readiness checks every selected profile and permits lazy loading',()=>{
  for(const model of Object.values(MODELS)) {
    const available={modelHash:model.sha256,pipeline:model.pipeline,threshold:model.threshold,uncertainThreshold:model.uncertainThreshold,status:'available'};
    const health={status:'ready',protocolVersion:2,availableModels:{[model.id]:available}};
    assert.equal(backendCompatibility(health,model).ready,true);
    for(const change of [{modelHash:'old'},{pipeline:'old'},{threshold:.123},{uncertainThreshold:-1},{status:'loading'}]) {
      assert.equal(backendCompatibility({...health,availableModels:{[model.id]:{...available,...change}}},model).ready,false);
    }
    assert.equal(backendCompatibility({...health,availableModels:{}},model).ready,false);
  }
  assert.equal(backendCompatibility(undefined,Object.values(MODELS)[0]).ready,false);
});

test('late results cannot attach to detached or recycled Reddit image elements',()=>{
  const image={isConnected:true,currentSrc:'https://preview.redd.it/first.png',src:'https://preview.redd.it/fallback.png'};
  assert.equal(sameImage(image,image.currentSrc),true);
  assert.equal(sameImage({...image,currentSrc:'https://preview.redd.it/second.png'},image.currentSrc),false);
  assert.equal(sameImage({...image,isConnected:false},image.currentSrc),false);
  assert.equal(sameImage({...image,currentSrc:''},image.src),true);
});
