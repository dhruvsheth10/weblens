import {getSettings,sourceForModelSwitch,type Settings} from './settings';
import {MODELS,modelConfig,type ModelId} from '../detection/model';
import type {ImageSource} from './images';
import {SOURCE_HELP,type PageStatus} from './presentation';
const enabled=document.querySelector<HTMLInputElement>('#enabled')!,feed=document.querySelector<HTMLInputElement>('#feed')!,source=document.querySelector<HTMLSelectElement>('#source')!,model=document.querySelector<HTMLSelectElement>('#model')!;
const panel=document.querySelector<HTMLElement>('#scanner-state')!,label=document.querySelector<HTMLElement>('#scanner-label')!,detail=document.querySelector<HTMLElement>('#scanner-detail')!,counts=document.querySelector<HTMLElement>('#scanner-counts')!,notice=document.querySelector<HTMLElement>('#status')!;
model.replaceChildren(...Object.values(MODELS).map(config=>{const option=document.createElement('option');option.value=config.id;option.textContent=config.name;return option;}));
let settings=await getSettings();
function sync(){
  enabled.checked=settings.enabled;feed.checked=settings.feed;feed.disabled=!settings.enabled;
  document.querySelector('#feed-help')!.textContent=settings.enabled?'Check images as you scroll':'Turn on Run Reddit Lens first';
  source.value=settings.imageSource;model.value=settings.model;
  source.querySelector<HTMLOptionElement>('option[value="preview-first"]')!.disabled=modelConfig(settings.model).uncertainThreshold===undefined;
  document.querySelector('#source-help')!.textContent=SOURCE_HELP[settings.imageSource];
}
function display(mode:string,title:string,text:string,status?:PageStatus){
  panel.dataset.mode=mode;if(label.textContent!==title)label.textContent=title;if(detail.textContent!==text)detail.textContent=text;
  const summary=status?`${status.scanned} scanned · ${status.ai} AI signals${status.errors ? ` · ${status.errors} failed` : ''}`:'';
  if(counts.textContent!==summary)counts.textContent=summary;
}
let refreshing=false;
async function refresh(){
  if(refreshing)return;refreshing=true;
  try{
    const latest=await getSettings();if(JSON.stringify(latest)!==JSON.stringify(settings)){settings=latest;sync();}
    const [tab]=await chrome.tabs.query({active:true,currentWindow:true});
    let page:PageStatus|undefined;
    // Query the tab ID only. Content-script access does not grant tabs.url;
    // our own status listener exists only on the Reddit pages in the manifest.
    if(tab?.id!==undefined)try{page=await chrome.tabs.sendMessage(tab.id,{type:'SCANNER_STATUS'});}catch{}
    if(!settings.enabled){display('paused','Paused','Image scanning is off. Turn on Run Reddit Lens to begin.',page);return;}
    if(page){display(page.mode,page.label,page.detail,page);return;}
    const health=await chrome.runtime.sendMessage({target:'background',type:'BACKEND_STATUS'});
    if(!health?.ready){display('error','Local detector offline',health?.error??'Start npm run dev, then reload the extension and Reddit tab.');return;}
    display('ready','Enabled · waiting for Reddit','Open or reload Reddit to connect its live status panel and image badges.');
  }catch{display('error','Connection unavailable','Reload the extension and your Reddit tab.');}
  finally{refreshing=false;}
}
async function persist(patch:Partial<Settings>){
  try{settings={...await getSettings(),...patch};await chrome.storage.local.set({settings});sync();notice.textContent='Settings saved.';void refresh();}
  catch{notice.textContent='Settings could not be saved. Reload the extension.';}
}
enabled.addEventListener('change',()=>void persist({enabled:enabled.checked}));
feed.addEventListener('change',()=>void persist({feed:feed.checked}));
source.addEventListener('change',()=>void persist({imageSource:source.value as ImageSource}));
model.addEventListener('change',async()=>{
  const modelId=model.value as ModelId,current=await getSettings();
  await persist({model:modelId,threshold:modelConfig(modelId).threshold,imageSource:sourceForModelSwitch(current.imageSource,modelId)});
  await chrome.storage.session.remove('results');notice.textContent=`${modelConfig(modelId).name} selected.`;
});
document.querySelector('#lab')!.addEventListener('click',()=>void chrome.runtime.openOptionsPage());
chrome.storage.onChanged.addListener((changes,area)=>{if(area==='local'&&changes.settings)void refresh();});
sync();void refresh();setInterval(()=>void refresh(),1000);
