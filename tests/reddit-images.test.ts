import test from 'node:test';
import assert from 'node:assert/strict';
import {originalCandidate, redditMediaUrl, imageInput, fetchImage} from '../src/extension/images';
import {MAX_BYTES} from '../src/detection/model';
const preview='https://preview.redd.it/abc123def456.png?width=640&auto=webp&s=123';
const original='https://i.redd.it/abc123def456.png';
function response(url:string, type='image/png', body:BodyInit=new Uint8Array([1,2,3]), status=200, headers:Record<string,string>={}) {
  const value=new Response(body,{status,headers:{'content-type':type,...headers}});
  Object.defineProperty(value,'url',{value:url});return value;
}
test('original candidates strip preview codecs without rewriting external or arbitrary paths',()=>{
  assert.equal(originalCandidate(preview),original);
  assert.equal(originalCandidate(preview,'https://i.redd.it/abc123def456.jpg'),'https://i.redd.it/abc123def456.jpg');
  assert.equal(originalCandidate(preview,'https://i.redd.it/unrelated123.png'),original);
  assert.equal(originalCandidate('https://external-preview.redd.it/abc123def456.png'),undefined);
  assert.equal(originalCandidate('https://preview.redd.it/directory/abc123def456.png'),undefined);
  assert.equal(originalCandidate('https://evil.test/abc123def456.png'),undefined);
  assert.equal(originalCandidate('https://preview.redd.it/caption-v0-abc123def456.webp'),undefined);
});
test('modern slugs and gallery links resolve only the matching media ID',()=>{
  const modern='https://preview.redd.it/some-long-title-v0-abc123def456.jpeg?width=640&format=pjpg&auto=webp&s=signed';
  const wrapped='https://www.reddit.com/media?url='+encodeURIComponent(original);
  assert.equal(originalCandidate(modern),'https://i.redd.it/abc123def456.jpeg');
  assert.equal(originalCandidate(modern,['https://i.redd.it/unrelated123.jpg',wrapped]),original);
  assert.equal(originalCandidate('https://preview.redd.it/caption-v0-abc123def456.webp',[original]),original);
  assert.equal(redditMediaUrl(wrapped),original);
  assert.equal(redditMediaUrl('https://www.reddit.com/media?url=https://evil.test/x.png'),undefined);
  assert.equal(originalCandidate('https://preview.redd.it/not-an-id.jpeg'),undefined);
});
test('prefer-original requests original bytes and records actual safe response URL',async()=>{
  const requested:string[]=[];
  const fetcher:typeof fetch=async(input,init)=>{requested.push(String(input));assert.equal(init?.credentials,'omit');return response(original);};
  const result=await imageInput(preview,'prefer-original',undefined,fetcher);
  assert.deepEqual(requested,[original]);assert.equal(result.inputSource,'original');
  assert.equal(result.sourceUrl,original);assert.equal(result.downloadBytes,3);assert.equal(result.fallbackReason,undefined);
  assert.equal(result.contentType,'image/png');
});
test('failed original falls back to the displayed bytes and exposes the failure',async()=>{
  const requested:string[]=[];
  const fetcher:typeof fetch=async(input)=>{const url=String(input);requested.push(url);return response(url,url===original?'text/html':'image/webp',new Uint8Array([1]),url===original?404:200);};
  const result=await imageInput(preview,'prefer-original',undefined,fetcher);
  assert.deepEqual(requested,[original,preview]);assert.equal(result.inputSource,'displayed');
  assert.match(result.fallbackReason!,/404/);assert.equal(result.blob.type,'image/webp');
});
test('displayed mode and external previews issue exactly one request',async()=>{
  for(const url of [preview,'https://external-preview.redd.it/abc123def456.png']){
    const requested:string[]=[];
    await imageInput(url,'displayed',undefined,async(input)=>{requested.push(String(input));return response(String(input));});
    assert.deepEqual(requested,[url]);
  }
});
test('fetch refuses unsafe redirects, nonimages, empty bodies and excess byte declarations',async()=>{
  for(const bad of [response('https://evil.test/redirect.png'),response(original,'text/html'),response(original,'image/png',new Uint8Array()),response(original,'image/png',new Uint8Array([1]),200,{'content-length':String(MAX_BYTES+1)})]){
    await assert.rejects(()=>fetchImage(original,async()=>bad));
  }
});
test('stream size limits apply even with no content-length',async()=>{
  let cancelled=false;
  const body=new ReadableStream({start(controller){controller.enqueue(new Uint8Array(MAX_BYTES+1));},cancel(){cancelled=true;}});
  await assert.rejects(()=>fetchImage(original,async()=>response(original,'image/png',body)),/20 MB/);
  assert.equal(cancelled,true);
});
