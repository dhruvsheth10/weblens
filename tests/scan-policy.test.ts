import test from 'node:test';
import assert from 'node:assert/strict';
import {scanImage,type Capture} from '../src/extension/scan';
import {modelConfig,type Detection} from '../src/detection/model';

const model='community-ferret-preview-v1' as const;
const displayed='https://preview.redd.it/a-title-v0-abc123def456.png?width=640&auto=webp&s=signed';
const original='https://i.redd.it/abc123def456.jpg';
function observation(score:number,semanticScore=score):Detection {
  const profile=modelConfig(model);
  return {model,modelHash:profile.sha256,pipeline:profile.pipeline,score,logit:Math.log(score/(1-score)),semanticScore,
    width:320,height:320,elapsedMs:10,engine:'cpu',variant:'original',uncertainThreshold:profile.uncertainThreshold};
}
function scenario(previewScore:number,originalScore=.8,failedOriginal=false,semanticScore=previewScore) {
  const requests:string[]=[];
  const fetcher:typeof fetch=async(input)=>{
    const url=String(input);requests.push(url);
    const response=new Response(new Uint8Array([url===original?2:1]),{status:url===original&&failedOriginal?403:200,headers:{'content-type':'image/png'}});
    Object.defineProperty(response,'url',{value:url});return response;
  };
  return {requests,fetcher,analyze:async(blob:Blob)=>observation(new Uint8Array(await blob.arrayBuffer())[0]===2?originalScore:previewScore,new Uint8Array(await blob.arrayBuffer())[0]===2?originalScore:semanticScore)};
}
const options={url:displayed,model,source:'preview-first' as const,threshold:.5,context:{candidates:[original]}};
test('strong and unmarked previews need no original; the .5 boundary is strong',async()=>{
  for(const score of [.8,.5,.01]) {
    const deps=scenario(score);const result=await scanImage(options,deps);
    assert.deepEqual(deps.requests,[displayed]);assert.equal(result.score,score);assert.equal(result.input?.escalated,false);
  }
});
test('uncertain preview escalates and exposes interim result before original inference',async()=>{
  const deps=scenario(.1);const previews:number[]=[];
  const result=await scanImage(options,{...deps,onPreview:r=>previews.push(r.score)});
  assert.deepEqual(previews,[.1]);assert.deepEqual(deps.requests,[displayed,original]);assert.equal(result.score,.8);
  assert.equal(result.input?.inputSource,'original');assert.equal(result.input?.previewScore,.1);assert.equal(result.input?.originalStatus,'fetched');
});
test('observable progress follows actual operations without changing the final detector result',async()=>{
  const deps=scenario(.1);const stages:string[]=[];
  const result=await scanImage({...options,record:true},{...deps,onProgress:step=>stages.push(`${step.phase}:${step.role}`),record:async()=>({id:'capture',sessionId:'session'})});
  assert.deepEqual(stages,['fetching:displayed','analyzing:displayed','fetching:original','analyzing:original','recording:original']);
  assert.equal(result.score,.8);assert.deepEqual(deps.requests,[displayed,original]);
});
test('semantic uncertainty prevents a weak pixel-only signal forcing an original request',async()=>{
  const deps=scenario(.4,.8,false,.001);const result=await scanImage(options,deps);
  assert.deepEqual(deps.requests,[displayed]);assert.equal(result.input?.escalated,false);
});
test('original 403 keeps the analyzed preview and records its failure status',async()=>{
  const deps=scenario(.1,.8,true);const result=await scanImage(options,deps);
  assert.equal(result.score,.1);assert.equal(result.input?.originalStatus,'failed');assert.match(result.input?.fallbackReason!,/403/);
  assert.equal(result.input?.attempts?.[1].status,403);assert.equal(result.input?.downloadBytes,1);
});
test('external preview uncertainty reports an unresolved original without fabricating a URL',async()=>{
  const deps=scenario(.1);const url='https://external-preview.redd.it/abc123def456.png';
  const result=await scanImage({...options,url},deps);
  assert.deepEqual(deps.requests,[url]);assert.equal(result.input?.originalStatus,'unresolved');assert.match(result.input?.fallbackReason!,/No matching original/);
});
test('record mode collects originals even for unmarked previews but keeps the browsing verdict',async()=>{
  const deps=scenario(.01);let saved:Capture|undefined;
  const result=await scanImage({...options,record:true},{...deps,record:async capture=>{saved=capture;return{id:'capture',sessionId:'session'};}});
  assert.deepEqual(deps.requests,[displayed,original]);assert.equal(saved?.original?.prediction.score,.8);
  assert.equal(result.score,.01);assert.equal(result.input?.sourceUrl,displayed);assert.equal(result.recording?.id,'capture');
});
test('a recording failure does not turn successful inference into an unavailable badge',async()=>{
  const deps=scenario(.8);const result=await scanImage({...options,record:true},{...deps,record:async()=>{throw new Error('Disk full');}});
  assert.equal(result.score,.8);assert.equal(result.recordingError,'Disk full');
});
test('already-original inputs reuse their bytes in record mode',async()=>{
  const deps=scenario(.1);let saved:Capture|undefined;
  const result=await scanImage({...options,url:original,record:true},{...deps,record:async capture=>{saved=capture;return{id:'capture',sessionId:'session'};}});
  assert.deepEqual(deps.requests,[original]);assert.equal(saved?.preview.blob,saved?.original?.blob);assert.equal(result.input?.originalStatus,'same-as-displayed');
});
test('original-first 403 falls back to displayed input and recording does not retry the failed URL',async()=>{
  const deps=scenario(.1,.8,true);const result=await scanImage({...options,source:'prefer-original',record:true},deps);
  assert.deepEqual(deps.requests,[original,displayed]);assert.equal(result.score,.1);assert.equal(result.input?.originalStatus,'failed');
});

test('failed recording preview preserves the original verdict and surfaces a retryable recording error',async()=>{
  const deps=scenario(.1,.8);let saved=false;
  const result=await scanImage({...options,source:'prefer-original',record:true},{...deps,fetcher:async(input,init)=>{
    if(String(input)===displayed)throw new Error('Preview download failed');
    return deps.fetcher(input,init);
  },record:async()=>{saved=true;return {id:'capture',sessionId:'session'};}});
  assert.equal(result.score,.8);assert.equal(saved,false);assert.equal(result.recording,undefined);
  assert.match(result.recordingError!,/Preview download failed/);
  assert.equal(result.input?.sourceUrl,original);
});

test('record mode without a connected saver does not claim success',async()=>{
  const result=await scanImage({...options,record:true},scenario(.8));
  assert.equal(result.score,.8);assert.equal(result.recording,undefined);
  assert.match(result.recordingError!,/recording is unavailable/);
});
