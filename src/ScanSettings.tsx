import {useEffect,useState} from 'react';
import {modelConfig} from './detection/model';
import type {Settings} from './extension/settings';
import {SOURCE_HELP} from './extension/presentation';
type Props={settings:Settings;save:(patch:Partial<Settings>)=>Promise<void>;extension:boolean};
export function ScanSettings({settings,save,extension}:Props) {
  const [stats,setStats]=useState<{count:number;failures:number;originals:number;unresolved:number;http403:number;median:number;p95:number}|null>(null);
  useEffect(()=>{if(!extension)return;
    const refresh=async()=>{
      const events=(await chrome.storage.session.get('scanEvents')).scanEvents??[];
      const times=events.filter((e:{ok:boolean;visibleMs?:number})=>e.ok&&Number.isFinite(e.visibleMs)).map((e:{visibleMs:number})=>e.visibleMs).sort((a:number,b:number)=>a-b);
      setStats({count:events.length,failures:events.filter((e:{ok:boolean})=>!e.ok).length,originals:events.filter((e:{input?:{originalStatus:string}})=>e.input?.originalStatus==='fetched').length,
        unresolved:events.filter((e:{input?:{originalStatus:string}})=>e.input?.originalStatus==='unresolved').length,
        http403:events.filter((e:{fetchStatus?:number;input?:{attempts?:{status?:number}[]}})=>e.fetchStatus===403||e.input?.attempts?.some(a=>a.status===403)).length,
        median:times[Math.floor(times.length/2)]??0,p95:times[Math.floor((times.length-1)*.95)]??0});
    };void refresh();const timer=setInterval(()=>void refresh(),5000);return()=>clearInterval(timer);
  },[extension]);
  return <>
    <label className="source-choice">Image source<select disabled={!extension} value={settings.imageSource} onChange={e=>void save({imageSource:e.target.value as Settings['imageSource']})}>
      <option value="preview-first" disabled={modelConfig(settings.model).uncertainThreshold===undefined}>Preview first · original for uncertain signals</option>
      <option value="prefer-original">Original first · preview fallback</option><option value="displayed">Displayed image only</option>
    </select></label>
    <p className="subtle">{SOURCE_HELP[settings.imageSource]}</p>
    {stats&&<details className="diagnostics"><summary>Fetch diagnostics</summary><p className="subtle" role="status">{stats.count} recent scan requests · {stats.failures} failed · {stats.originals} originals fetched · {stats.unresolved} unresolved originals · {stats.http403} HTTP 403<br/>Median {stats.median} ms / p95 {stats.p95} ms from visibility to final result.</p>
      <button className="text-button" onClick={()=>void chrome.storage.session.get('scanEvents').then(value=>{
        const url=URL.createObjectURL(new Blob([JSON.stringify(value.scanEvents??[],null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='reddit-lens-fetch-events.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      })}>Export fetch diagnostics</button></details>}
  </>;
}
