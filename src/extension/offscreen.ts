import {analyze} from '../detection/client';
import {ImageFetchError} from './images';
import {scanImage} from './scan';
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.target !== 'offscreen' || message.type !== 'ANALYZE_URL' || sender.id !== chrome.runtime.id || sender.tab) return;
  void (async () => {
    try {
      if (!['preview-first','prefer-original','displayed'].includes(message.imageSource)) throw new Error('Unknown image source policy.');
      const result=await scanImage({url:message.url,source:message.imageSource,model:message.model,threshold:message.threshold,context:message.context,record:false},
        {analyze:(blob,model)=>analyze(blob,'original',model),onProgress:progress=>{
          void chrome.runtime.sendMessage({target:'background',type:'SCAN_PROGRESS',requestId:message.requestId,progress}).catch(()=>{});
        },onPreview:preview=>{
          void chrome.runtime.sendMessage({target:'background',type:'PREVIEW_READY',requestId:message.requestId,result:preview}).catch(()=>{});
        }});
      respond({result});
    }
    catch (error) {respond({error: error instanceof Error ? error.message : String(error),fetchStatus:error instanceof ImageFetchError?error.status:undefined});}
  })(); return true;
});
