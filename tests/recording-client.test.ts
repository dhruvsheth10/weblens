import test from 'node:test';
import assert from 'node:assert/strict';
import {recordCapture} from '../src/recordings';
import {MODEL,type Detection} from '../src/detection/model';
import type {Capture} from '../src/extension/scan';
const session='a'.repeat(32);
const result:Detection={model:MODEL.id,modelHash:MODEL.sha256,pipeline:MODEL.pipeline,score:.2,logit:Math.log(.2/.8),width:256,height:256,engine:'cpu',variant:'original',elapsedMs:1};
const capture:Capture={options:{url:'https://preview.redd.it/abc123def456.png',source:'preview-first',model:MODEL.id,threshold:.5},preview:{blob:new Blob([new Uint8Array([1])],{type:'image/png'}),url:'https://preview.redd.it/abc123def456.png',prediction:result},result};
async function failure(active:boolean,offline=false) {
  const oldFetch=globalThis.fetch,oldChrome=globalThis.chrome;
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,'location');
  const writes:unknown[]=[];let calls=0;
  Object.defineProperty(globalThis,'location',{configurable:true,value:{protocol:'chrome-extension:'}});
  // Match the restricted API surface of an actual offscreen document.
  globalThis.chrome={runtime:{sendMessage:async(value:unknown)=>{writes.push(value);return{active:false};}}} as unknown as typeof chrome;
  globalThis.fetch=async()=>{
    if(++calls===1)return new Response(JSON.stringify({detail:'Recording limit reached'}),{status:422});
    if(offline)throw new Error('Offline');
    return new Response(JSON.stringify({sessions:[{id:session,active:Number(active)}]}));
  };
  try{await assert.rejects(recordCapture(session,capture),/Recording limit reached/);return writes;}
  finally{globalThis.fetch=oldFetch;globalThis.chrome=oldChrome;if(descriptor)Object.defineProperty(globalThis,'location',descriptor);else Reflect.deleteProperty(globalThis,'location');}
}
test('a stopped recording clears the session and future record-only acquisition',async()=>{
  assert.deepEqual(await failure(false),[{target:'background',type:'RECORDING_STOPPED',session}]);
});
test('a failed capture in an active session keeps the users recording choice',async()=>{
  assert.deepEqual(await failure(true),[]);
});
test('an unavailable status endpoint keeps intent and preserves the original failure',async()=>{
  assert.deepEqual(await failure(false,true),[]);
});
