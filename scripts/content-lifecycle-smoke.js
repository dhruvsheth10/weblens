// Browser integration regression in a disposable local lab tab.
// Mock transport + synthetic DOM only; no detector or Reddit requests.
(async () => {
  const {MODELS}=await import('/src/detection/model.ts');
  const model=MODELS['community-ferret-preview-v1'];
  const settings={settingsVersion:4,enabled:true,feed:true,model:model.id,threshold:.5,imageSource:'displayed'};
  const listeners=[],requests=[];
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const until=async predicate=>{for(let i=0;i<100;i++){if(predicate())return;await wait(30);}throw new Error('DOM regression timed out');};
  const canvas=document.createElement('canvas');canvas.width=canvas.height=200;
  const context=canvas.getContext('2d');context.fillStyle='white';context.fillRect(0,0,200,200);
  const first=canvas.toDataURL();context.fillStyle='black';context.fillRect(0,0,200,200);const second=canvas.toDataURL();
  const article=document.createElement('article'),img=document.createElement('img');
  img.style.cssText='position:fixed;left:300px;top:100px;width:200px;height:200px';
  article.append(img);document.body.append(article);img.src=first;await img.decode();
  let currentUrl='https://preview.redd.it/abcdef123456.png';
  Object.defineProperty(img,'currentSrc',{get:()=>currentUrl});
  const result=score=>({model:model.id,modelHash:model.sha256,pipeline:model.pipeline,score,logit:Math.log(score/(1-score)),width:200,height:200,elapsedMs:1,engine:'cpu',variant:'original'});
  const stub={
    runtime:{id:'local-dom-test',onMessage:{addListener:f=>listeners.push(f)},sendMessage:async message=>{
      if(message.type==='BACKEND_STATUS')return {ready:true};
      if(message.type==='SCAN_IMAGE')return new Promise(resolve=>requests.push({message,resolve}));
      return {};
    }},
    storage:{local:{get:async()=>({settings})},onChanged:{addListener(){}}}
  };
  Object.defineProperty(window,'chrome',{value:stub,configurable:true});
  await import('/src/extension/content.ts');
  await until(()=>requests.length===1);
  const old=requests[0];
  currentUrl='https://preview.redd.it/uvwxyz123456.png';img.src=second;await img.decode();await wait(50);
  for(const listener of listeners)listener({type:'PREVIEW_READY',clientRequestId:old.message.clientRequestId,result:result(.99)},{id:stub.runtime.id},()=>{});
  const shadow=document.querySelector('#reddit-lens-overlay-root').shadowRoot;
  if(shadow.querySelector('.frame[data-state=ai]'))throw new Error('Stale preview attached to the replacement image');
  old.resolve({result:result(.99)});
  await until(()=>requests.length===2);requests[1].resolve({result:result(.01)});
  await until(()=>!!shadow.querySelector('.frame[data-state=unmarked]'));
  article.remove();await wait(450);document.body.append(article);
  await until(()=>requests.length===3);requests[2].resolve({result:result(.01)});
  await until(()=>!!shadow.querySelector('.frame[data-state=unmarked]'));
  return {kind:'synthetic-content-lifecycle-regression',passed:true,stalePreviewRejected:true,replacementRescanned:true,reinsertedImageRescanned:true,transportRequests:requests.length,note:'Reload the disposable tab to remove mocks/listeners. No accuracy claim.'};
})();
