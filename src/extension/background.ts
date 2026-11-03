import {isRedditImageUrl, modelConfig, type Detection, type ModelId} from '../detection/model';
import {getSettings,type Settings} from './settings';
import type {PageStatus} from './presentation';
import {backendCompatibility} from './health';
let backendStatus:{at:number;health:unknown}|undefined;
let checkingBackend:Promise<unknown>|undefined;
async function backendReady(force=false) {
  const settings=await getSettings(),model=modelConfig(settings.model);
  if(!force&&backendStatus&&Date.now()-backendStatus.at<4500)return backendCompatibility(backendStatus.health,model);
  checkingBackend??=(async()=>{
    let health:unknown;
    try{const response=await fetch('http://127.0.0.1:8000/api/health',{signal:AbortSignal.timeout(3000)});health=response.ok?await response.json():undefined;
      if(response.ok){backendStatus={at:Date.now(),health};return health;}
    }catch{}
    backendStatus={at:Date.now(),health};return health;
  })().finally(()=>{checkingBackend=undefined;});return backendCompatibility(await checkingBackend,model);
}
let creating: Promise<void> | undefined;
const pending = new Map<string, Promise<Detection>>();
let storageWrites=Promise.resolve();
function writeSession(update:()=>Promise<void>) {
  const next=storageWrites.then(update,update);
  storageWrites=next.catch(()=>{});return next;
}
function logEvent(event:Record<string,unknown>) {
  return writeSession(async()=>{
    const history=(await chrome.storage.session.get('scanEvents')).scanEvents??[];
    history.push(event);await chrome.storage.session.set({scanEvents:history.slice(-500)});
  });
}
async function offscreen() {
  const contexts = await chrome.runtime.getContexts({contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT]});
  if (contexts.length) return;
  creating ??= chrome.offscreen.createDocument({url: 'offscreen.html', reasons: [chrome.offscreen.Reason.BLOBS, chrome.offscreen.Reason.WORKERS], justification: 'Fetch Reddit image bytes and submit them to the selected local detector.'}).finally(() => {creating = undefined;});
  await creating;
}
import {originalCandidate, type PostContext} from './images';
const previews=new Map<string,{tabId:number;url:string;model:string;source:string;threshold:number;session?:string;clientRequestId?:string}>();
async function scan(url: string, settings:Settings, context:PostContext, tabId:number,clientRequestId?:string) {
  const model = modelConfig(settings.model);
  const originalUrl = originalCandidate(url, context.candidates);
  const key = `fetch-v5:${model.sha256}:${model.pipeline}:${settings.imageSource}:${settings.threshold}:${settings.recordingSession??''}:${settings.recordingSession?context.postUrl??'':''}:${originalUrl ?? ''}:${url}`;
  const cached = (await chrome.storage.session.get('results')).results ?? {};
  if (cached[key]) return cached[key] as Detection;
  if (pending.has(key)) return pending.get(key)!;
  if (pending.size >= 24) throw new Error('Scan queue full. Try again after current images finish.');
  const task = (async () => {
    await offscreen();
    const requestId=crypto.randomUUID();previews.set(requestId,{tabId,url,model:settings.model,source:settings.imageSource,threshold:settings.threshold,session:settings.recordingSession,clientRequestId});
    let response;
    try {response = await chrome.runtime.sendMessage({target: 'offscreen', type: 'ANALYZE_URL',requestId,url,context,imageSource:settings.imageSource,threshold:settings.threshold,model:settings.model});}
    finally {previews.delete(requestId);}
    if (!response?.result) throw Object.assign(new Error(response?.error ?? 'Inference document did not respond.'),{fetchStatus:response?.fetchStatus});
    // Serialize read/modify/write across tabs. Failed recordings remain retryable.
    if(!response.result.recordingError)await writeSession(async()=>{
      const latest = (await chrome.storage.session.get('results')).results ?? {};
      latest[key] = response.result;
      const entries = Object.entries(latest).slice(-200);
      await chrome.storage.session.set({results: Object.fromEntries(entries)});
    });
    return response.result as Detection;
  })().finally(() => pending.delete(key));
  pending.set(key, task); return task;
}
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if(message?.target==='background'&&message.type==='BACKEND_STATUS'&&sender.id===chrome.runtime.id){
    void backendReady(message.force===true).then(respond);return true;
  }
  if(message?.target==='background'&&message.type==='OPEN_OPTIONS'&&sender.id===chrome.runtime.id){
    void chrome.runtime.openOptionsPage().then(()=>respond({opened:true})).catch(()=>respond({opened:false}));return true;
  }
  if(message?.target==='background'&&message.type==='PAGE_STATUS'&&sender.id===chrome.runtime.id&&sender.tab?.id!==undefined&&/^https:\/\/(www\.|old\.|new\.)?reddit\.com\//.test(sender.url??'')){
    const status=message.status as PageStatus;
    const text=status?.mode==='paused'?'OFF':status?.mode==='scope'?'POST':status?.mode==='error'?'ERR':status?.mode==='scanning'?'SCAN':status?.mode==='ready'?'ON':'…';
    const tabId=sender.tab.id;void Promise.all([
      chrome.action.setBadgeText({tabId,text}),chrome.action.setBadgeBackgroundColor({tabId,color:text==='ERR'?'#a13850':text==='SCAN'?'#6550ad':text==='ON'?'#14776f':'#656579'}),
      chrome.action.setTitle({tabId,title:`Reddit Lens · ${status?.label??'Starting'} · ${status?.scanned??0} scanned · ${status?.ai??0} AI signals`}),
    ]).catch(()=>{});return;
  }
  if(message?.target==='background' && ['RECORDING_ACTIVE','RECORDING_STOPPED'].includes(message.type)){respond({active:false});return;}
  if(message?.target==='background' && ['PREVIEW_READY','SCAN_PROGRESS'].includes(message.type) && sender.id===chrome.runtime.id && !sender.tab) {
    const target=previews.get(message.requestId);
    if(target)void chrome.tabs.sendMessage(target.tabId,{type:message.type,...target,result:message.result,progress:message.progress}).catch(()=>{});
    return;
  }
  if (message?.target !== 'background' || message.type !== 'SCAN_IMAGE') return;
  const senderUrl = sender.url ?? sender.tab?.url ?? '';
  const permitted = /^https:\/\/(www\.|old\.|new\.)?reddit\.com\//.test(senderUrl);
  if (sender.id !== chrome.runtime.id || !sender.tab || !permitted || !isRedditImageUrl(message.url)) {respond({error: 'Unsupported image or sender.'}); return;}
  void (async () => {
    try {
      const settings = await getSettings();
      if (!settings.enabled) throw new Error('Reddit scanning is paused.');
      if (!settings.feed && !/\/comments\//.test(senderUrl)) throw new Error('Feed scanning is disabled.');
      if (message.model !== settings.model) throw new Error('Detection model changed. Retry with the current model.');
      const started=Date.now();
      const context=message.context && typeof message.context==='object'?message.context:{};
      const result=await scan(message.url,settings,context,sender.tab!.id!,typeof message.clientRequestId==='string'?message.clientRequestId:undefined);
      await logEvent({at:new Date().toISOString(),postUrl:context.postUrl,url:message.url,model:settings.model,ok:true,
        visibleMs:Number.isFinite(message.visibleAt)?Math.max(0,Date.now()-message.visibleAt):Date.now()-started,input:result.input,recording:result.recording,recordingError:result.recordingError});
      respond({result});
    } catch (error) {
      const value=error instanceof Error ? error.message : String(error);
      await logEvent({at:new Date().toISOString(),url:message.url,postUrl:message.context?.postUrl,ok:false,error:value,fetchStatus:(error as {fetchStatus?:number})?.fetchStatus}).catch(()=>{});
      respond({error:value});
    }
  })();
  return true;
});

// Persist the one-time default migration and discard any old collection session.
void getSettings().then(async settings=>{
  const stored=(await chrome.storage.local.get('settings')).settings;
  if(stored?.settingsVersion!==settings.settingsVersion||stored?.recordingSession)
    await chrome.storage.local.set({settings});
}).catch(()=>{});
