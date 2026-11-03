import test from 'node:test';
import assert from 'node:assert/strict';
import {getSettings,DEFAULT_SETTINGS,sourceForModelSwitch} from '../src/extension/settings';
import {modelConfig} from '../src/detection/model';
async function settings(value:object) {
  const previous=globalThis.chrome;
  globalThis.chrome={storage:{local:{get:async()=>({settings:value})}}} as unknown as typeof chrome;
  try{return await getSettings();}finally{globalThis.chrome=previous;}
}
test('new installs default to Sherlock 2 at its packaged cutoff, displayed images and no recording',async()=>{
  assert.equal(DEFAULT_SETTINGS.model,'reddit-sherlock-2');
  assert.equal(DEFAULT_SETTINGS.imageSource,'displayed');
  assert.equal(DEFAULT_SETTINGS.threshold,modelConfig('reddit-sherlock-2').threshold);
  assert.deepEqual(await settings({}),{...DEFAULT_SETTINGS,recordingSession:undefined});
});
test('retired standalone choices migrate to the default and reset incompatible cutoffs',async()=>{
  const stored={settingsVersion:2,model:'ferretnet-b-v1',enabled:true,feed:true,threshold:.8304235935211183,imageSource:'prefer-original'};
  assert.deepEqual(await settings(stored),{...stored,model:DEFAULT_SETTINGS.model,threshold:.5,imageSource:'displayed',settingsVersion:4,recordingSession:undefined});
});
test('unsupported uncertainty policies and invalid recording IDs are safely normalized',async()=>{
  const result=await settings({settingsVersion:4,model:'mirror-h-v1',imageSource:'preview-first',threshold:NaN,recordingSession:'../../outside'});
  assert.equal(result.imageSource,'prefer-original');assert.equal(result.threshold,modelConfig('mirror-h-v1').threshold);assert.equal(result.recordingSession,undefined);
});
test('prototype names are not treated as model identities',async()=>{
  for(const model of ['__proto__','constructor','toString']) {
    assert.equal(modelConfig(model).id,DEFAULT_SETTINGS.model);
    assert.equal((await settings({settingsVersion:4,model})).model,DEFAULT_SETTINGS.model);
  }
});

test('the experimental Reddit student selects displayed previews while explicit source overrides persist',async()=>{
  assert.equal(sourceForModelSwitch('preview-first','reddit-student-last2-v1'),'displayed');
  assert.equal(sourceForModelSwitch('prefer-original','reddit-student-last2-v1'),'displayed');
  assert.equal((await settings({settingsVersion:4,model:'reddit-student-last2-v1',threshold:.5})).imageSource,'displayed');
  assert.equal((await settings({settingsVersion:4,model:'reddit-student-last2-v1',threshold:.5,imageSource:'prefer-original'})).imageSource,'prefer-original');
});

test('old preferences migrate once to Sherlock 2 and valid saved collection sessions are ignored',async()=>{
  const old=await settings({settingsVersion:3,model:'community-ferret-preview-v1',threshold:.15,imageSource:'prefer-original',enabled:true,feed:true,recordingSession:'a'.repeat(32)});
  assert.deepEqual(old,{...DEFAULT_SETTINGS,enabled:true,feed:true,recordingSession:undefined});
  const current=await settings({...old,model:'community-preview-v1',threshold:.6,imageSource:'preview-first',recordingSession:'b'.repeat(32)});
  assert.equal(current.model,'community-preview-v1');assert.equal(current.threshold,.6);assert.equal(current.imageSource,'preview-first');assert.equal(current.recordingSession,undefined);
});
