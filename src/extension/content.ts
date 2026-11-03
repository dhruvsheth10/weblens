import {isRedditImageUrl,modelConfig,type Detection} from '../detection/model';
import {getSettings,DEFAULT_SETTINGS,type Settings} from './settings';
import {postElement,postContext} from './images';
import {RedditOverlay,OVERLAY_ID} from './overlay';
import {detectedMark,summarizeResults,type ImageMark,type PageStatus} from './presentation';
import type {ScanProgress} from './scan';
import {sameImage} from './lifecycle';

let settings:Settings=DEFAULT_SETTINGS,loaded=false;
let backend:'checking'|'ready'|'offline'='checking',lastError='',fatalError='';
const queued=new Set<HTMLImageElement>(),visible=new Set<HTMLImageElement>(),tracked=new Set<HTMLImageElement>();
const visibleAt=new WeakMap<HTMLImageElement,number>();
const results=new Map<HTMLImageElement,{url:string;result:Detection}>();
const completed=new Map<string,ImageMark>();
let current:{id:string;image:HTMLImageElement;url:string;settings:Settings;started:number;phase:string;preview?:Detection;invalidated?:boolean}|undefined;
let lastPublished='';
const overlay=new RedditOverlay({toggle:()=>void toggle(),retry:()=>void retry(),settings:()=>void chrome.runtime.sendMessage({target:'background',type:'OPEN_OPTIONS'}).catch(()=>{fatalError='Reload this Reddit tab after reloading the extension.';update();})});
function allowedScope(){return settings.feed||/\/comments\//.test(location.pathname);}
function eligible(image:HTMLImageElement){return settings.enabled&&allowedScope()&&image.complete&&image.naturalWidth>=128&&image.naturalHeight>=128&&isRedditImageUrl(image.currentSrc||image.src)&&!!postElement(image);}
function status():PageStatus {
  let mode:PageStatus['mode']='ready',label='Running',detail='Watching for visible Reddit images.';
  if(fatalError){mode='error';label='Reload needed';detail=fatalError;}
  else if(!loaded){mode='starting';label='Starting';detail='Loading extension settings…';}
  else if(!settings.enabled){mode='paused';label='Paused';detail=current?'The current request is finishing. No new scans will start.':'Image scanning is off. Start it here or in the extension popup.';}
  else if(!allowedScope()){mode='scope';label='Feed scans off';detail='Opened posts are enabled. Enable feed scanning to check images as you scroll.';}
  else if(backend==='offline'){mode='error';label='Detector offline';detail=lastError||'Start the local detector with npm run dev.';}
  else if(backend==='checking'){mode='checking';label='Connecting';detail='Checking the local detector…';}
  else if(current){mode='scanning';label='Scanning';const seconds=Math.floor((Date.now()-current.started)/1000);detail=`${current.phase}${seconds>=2?` · ${seconds}s`:''}${queued.size?` · ${queued.size} waiting`:''}`;}
  else if(lastError){mode='error';label='Scan failed';detail=lastError;}
  return {mode,label,detail,...summarizeResults(completed.values()),queued:queued.size,recording:!!settings.recordingSession,model:modelConfig(settings.model).name};
}
function update(){
  const value=status(),serialized=JSON.stringify(value);overlay.schedule();
  if(serialized===lastPublished)return;lastPublished=serialized;overlay.update(value);
  void chrome.runtime.sendMessage({target:'background',type:'PAGE_STATUS',status:value}).catch(()=>{});
}
function show(image:HTMLImageElement,result:Detection,pending=false){
  if(!eligible(image)){overlay.remove(image);return;}
  overlay.mark(image,detectedMark(result,modelConfig(settings.model),settings.threshold,pending));
}
function fail(image:HTMLImageElement,url:string,message:string){
  if(!eligible(image)||(image.currentSrc||image.src)!==url)return;
  image.dataset.redditLensError=message;lastError=message;
  const mark:ImageMark={state:'error',label:'Scan failed',detail:'No detection result · retry available',title:message};
  completed.set(url,mark);overlay.mark(image,mark);update();
}
function matches(request:NonNullable<typeof current>){return !request.invalidated&&settings.enabled&&allowedScope()&&request.settings.model===settings.model&&request.settings.imageSource===settings.imageSource&&request.settings.threshold===settings.threshold&&request.settings.recordingSession===settings.recordingSession;}
function pump(){
  if(current||!settings.enabled||!allowedScope()||backend!=='ready'||document.hidden){update();return;}
  while(queued.size){
    const image=queued.values().next().value!;queued.delete(image);
    if(!image.isConnected||!visible.has(image)||!eligible(image))continue;
    const url=image.currentSrc||image.src,stored=results.get(image);
    if(stored?.url===url&&stored.result.model===settings.model){completed.set(url,detectedMark(stored.result,modelConfig(settings.model),settings.threshold));show(image,stored.result);continue;}
    const request={id:crypto.randomUUID(),image,url,settings:{...settings},started:Date.now(),phase:'Checking image…'};
    current=request;overlay.mark(image,{state:'scanning',label:'Scanning image…',detail:'Reddit Lens is checking this image',title:'Fetching image bytes and checking them with your local detector.',busy:true});update();
    void chrome.runtime.sendMessage({target:'background',type:'SCAN_IMAGE',url,context:postContext(image),visibleAt:visibleAt.get(image),model:request.settings.model,clientRequestId:request.id}).then(response=>{
      if(!matches(request)){if(image.isConnected&&eligible(image)&&visible.has(image))queued.add(image);return;}
      if(!sameImage(image,url))return;
      if(!response?.result){fail(image,url,response?.error??'No scanner response. Reload the extension and this Reddit tab.');return;}
      const expected=modelConfig(settings.model),result=response.result as Detection;
      if(result.model!==expected.id||result.modelHash!==expected.sha256||result.pipeline!==expected.pipeline){fail(image,url,'Detector version mismatch. Reload the extension and this Reddit tab.');return;}
      image.removeAttribute('data-reddit-lens-error');lastError='';results.set(image,{url,result});completed.set(url,detectedMark(result,expected,settings.threshold));show(image,result);
    }).catch(error=>{if(matches(request))fail(image,url,error instanceof Error?error.message:String(error));}).finally(()=>{if(current?.id===request.id)current=undefined;pump();});
    return;
  }
  update();
}
const observer=new IntersectionObserver(entries=>{
  for(const entry of entries){const image=entry.target as HTMLImageElement;
    if(entry.isIntersecting){const wasVisible=visible.has(image);visible.add(image);overlay.visibility(image,true);
      if(eligible(image)){if(!wasVisible)visibleAt.set(image,Date.now());queued.add(image);}
    }else{visible.delete(image);queued.delete(image);overlay.visibility(image,false);}
  }
  pump();
},{threshold:.25});
const observed=new WeakSet<HTMLImageElement>(),roots=new WeakSet<Node>();let scheduled=false;
function discover(root:Document|ShadowRoot=document,reobserve=false){
  if(!roots.has(root)){roots.add(root);mutation.observe(root,{childList:true,subtree:true,attributes:true,attributeFilter:['src','srcset']});}
  for(const image of root.querySelectorAll<HTMLImageElement>('img')){
    if(reobserve)observer.observe(image);
    if(!tracked.has(image)){tracked.add(image);observer.observe(image);}
    if(!observed.has(image)){observed.add(image);
      image.addEventListener('load',()=>{if(current?.image===image)current.invalidated=true;results.delete(image);overlay.remove(image);visibleAt.delete(image);observer.unobserve(image);observer.observe(image);});
    }
  }
  for(const node of root.querySelectorAll('*'))if(node.id!==OVERLAY_ID&&node.shadowRoot)discover(node.shadowRoot,reobserve);
  for(const image of tracked)if(!image.isConnected){tracked.delete(image);visible.delete(image);queued.delete(image);results.delete(image);overlay.remove(image);overlay.visibility(image,false);observer.unobserve(image);}
}
const mutation=new MutationObserver(records=>{
  for(const record of records)if(record.type==='attributes' && record.target instanceof HTMLImageElement){
    if(current?.image===record.target)current.invalidated=true;
    results.delete(record.target);overlay.remove(record.target);
  }
  if(scheduled)return;scheduled=true;setTimeout(()=>{scheduled=false;discover();overlay.schedule();},300);
});
function reobserve(){observer.disconnect();discover(document,true);}
let checking:Promise<void>|undefined;
function checkBackend(force=false){
  checking??=(async()=>{
    try{const response=await chrome.runtime.sendMessage({target:'background',type:'BACKEND_STATUS',force});const wasOffline=backend!=='ready';
      if(response?.ready){backend='ready';if(wasOffline){lastError='';reobserve();}pump();}
      else{backend='offline';lastError=response?.error??'Reload the extension and Reddit tab, and start npm run dev.';}
    }catch{backend='offline';lastError='Cannot reach the extension. Reload the extension and this Reddit tab.';}
    update();
  })().finally(()=>{checking=undefined;});return checking;
}
async function save(patch:Partial<Settings>){
  try{const latest=await getSettings();await chrome.storage.local.set({settings:{...latest,...patch}});}
  catch{fatalError='Settings could not be saved. Reload the extension and this Reddit tab.';update();}
}
async function toggle(){if(!settings.enabled)await save({enabled:true,feed:settings.feed||!/\/comments\//.test(location.pathname)});else if(!allowedScope())await save({feed:true});else await save({enabled:false});}
async function retry(){lastError='';for(const [url,value] of completed)if(value.state==='error')completed.delete(url);backend='checking';update();await checkBackend(true);reobserve();}
void getSettings().then(value=>{settings=value;loaded=true;discover();update();void checkBackend();}).catch(()=>{loaded=true;fatalError='Reload the extension and this Reddit tab to initialize scanning.';update();});
chrome.storage.onChanged.addListener((changes,area)=>{
  if(area!=='local'||!changes.settings)return;
  void getSettings().then(value=>{
    const changed=value.model!==settings.model||value.imageSource!==settings.imageSource||value.threshold!==settings.threshold||value.recordingSession!==settings.recordingSession;
    settings=value;queued.clear();
    if(changed){results.clear();completed.clear();lastError='';overlay.clear();}
    else if(!settings.enabled||!allowedScope())overlay.clear();
    else for(const [image,stored] of results)show(image,stored.result);
    reobserve();update();if(settings.enabled)void checkBackend();
  }).catch(()=>{fatalError='Reload this Reddit tab after reloading the extension.';update();});
});
chrome.runtime.onMessage.addListener((message,sender,respond)=>{
  if(sender.id!==chrome.runtime.id)return;
  if(message?.type==='SCANNER_STATUS'){respond(status());return;}
  if(!['PREVIEW_READY','SCAN_PROGRESS'].includes(message?.type)||!current||message.clientRequestId!==current.id||!matches(current)||!sameImage(current.image,current.url))return;
  if(message.type==='PREVIEW_READY'){
    const result=message.result as Detection,expected=modelConfig(settings.model);
    if(result?.model===expected.id&&result.modelHash===expected.sha256&&result.pipeline===expected.pipeline){current.preview=result;show(current.image,result,true);}
  }else{
    const progress=message.progress as ScanProgress;
    if(!progress||!['fetching','analyzing','recording'].includes(progress.phase))return;
    current.phase=progress.phase==='recording'?'Saving image pair locally…':`${progress.phase==='fetching'?'Fetching':'Analyzing'} ${progress.role==='original'?'original':'displayed image'}…`;
    if(current.preview){const mark=detectedMark(current.preview,modelConfig(settings.model),settings.threshold,true);mark.detail=current.phase;overlay.mark(current.image,mark);}
    else overlay.mark(current.image,{state:'scanning',label:'Scanning image…',detail:current.phase,title:'Analysis is in progress. No final detection result yet.',busy:true});
  }
  update();
});
let lastPath=location.pathname;
setInterval(()=>{
  if(lastPath!==location.pathname){lastPath=location.pathname;queued.clear();completed.clear();lastError='';overlay.clear();for(const [image,stored] of results)show(image,stored.result);reobserve();}
  if(!document.hidden)update();
},1000);
setInterval(()=>{if(!document.hidden&&settings.enabled)void checkBackend();},5000);
document.addEventListener('visibilitychange',()=>{queued.clear();if(!document.hidden){reobserve();if(settings.enabled)void checkBackend();}overlay.schedule();update();});
update();
