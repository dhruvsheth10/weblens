import {useEffect,useState} from 'react';
import {backendUrl,recordingRequest} from './recordings';

type Session={id:string;active:number;captures:number;bytes:number;created:number};
type RecordRow={id:string;label:string;category:string;evidence:string;metadata:{postUrl?:string;subreddit?:string;preview:{url:string;prediction:{score:number};width:number;height:number};original?:{url:string;prediction:{score:number};width:number;height:number};fetch?:{fallbackReason?:string;originalStatus?:string}}};
function saveJson(value:unknown,name:string) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function ReviewRow({row,session,onSaved}:{row:RecordRow;session:string;onSaved:()=>void}) {
  const [label,setLabel]=useState(row.label),[category,setCategory]=useState(row.category),[evidence,setEvidence]=useState(row.evidence);
  const [status,setStatus]=useState(''),[busy,setBusy]=useState(false);
  async function save() {
    setBusy(true);setStatus('');
    try {await recordingRequest(`/${session}/captures/${row.id}/label`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label,category,evidence})});setStatus('Review saved.');onSaved();}
    catch(error){setStatus(String(error instanceof Error?error.message:error));}finally{setBusy(false);}
  }
  return <article className="capture-row">
    <div className="capture-images">{(['preview','original'] as const).map(kind=><div key={kind}>
      {row.metadata[kind]?<img src={backendUrl(`/api/recordings/${session}/captures/${row.id}/${kind}`)} alt={`Recorded ${kind}`} loading="lazy"/>:<div className="capture-missing">Original unavailable</div>}
      <strong>{kind==='preview'?'Displayed image':'Original'}</strong><small>{row.metadata[kind]?`${row.metadata[kind]!.width} × ${row.metadata[kind]!.height} · score ${row.metadata[kind]!.prediction.score.toFixed(4)}`:row.metadata.fetch?.fallbackReason??'No matching original URL.'}</small>
    </div>)}</div>
    <div className="capture-review"><p>{row.metadata.subreddit??'Subreddit not supplied'} · {row.id.slice(0,8)}</p>
      {row.metadata.postUrl?<a href={row.metadata.postUrl} target="_blank" rel="noreferrer">Open source post ↗</a>:<span className="subtle">Post permalink unavailable</span>}
      <label>Verified origin<select value={label} onChange={e=>setLabel(e.target.value)}><option value="unknown">Unknown / unreviewed</option><option value="real">Verified real</option><option value="ai">Verified AI</option></select></label>
      <label>Category<input value={category} maxLength={128} onChange={e=>setCategory(e.target.value)} placeholder="Photo, screenshot, meme, artwork…"/></label>
      <label>Evidence<textarea value={evidence} maxLength={4000} onChange={e=>setEvidence(e.target.value)} placeholder="Original creator/file, generator record, or evidence URL"/></label>
      <button className="button secondary" disabled={busy || label!=='unknown'&&!evidence.trim()} onClick={()=>void save()}>{busy?'Saving…':'Save review'}</button>
      <p role="status" className="subtle">{status}</p>
    </div>
  </article>;
}
export function RecordingPanel() {
  const [sessions,setSessions]=useState<Session[]>([]),[selected,setSelected]=useState(''),[rows,setRows]=useState<RecordRow[]>([]),[rowsKey,setRowsKey]=useState('');
  const [offset,setOffset]=useState(0),[error,setError]=useState(''),[root,setRoot]=useState(''),[revision,setRevision]=useState(0),[loading,setLoading]=useState(false);
  useEffect(()=>{let current=true;
    void recordingRequest('').then(value=>{if(!current)return;setSessions(value.sessions);setRoot(value.root);setSelected(previous=>previous||value.sessions[0]?.id||'');}).catch(e=>{if(current)setError(e.message);});
    return()=>{current=false;};
  },[revision]);
  useEffect(()=>{let current=true;if(!selected){setRows([]);setLoading(false);return;}setLoading(true);
    void recordingRequest(`/${selected}/captures?offset=${offset}`).then(value=>{if(current){setRows(value.captures);setRowsKey(`${selected}:${offset}`);setError('');}}).catch(e=>{if(current)setError(e.message);}).finally(()=>{if(current)setLoading(false);});
    return()=>{current=false;};
  },[selected,offset,revision]);
  const active=sessions.find(s=>s.id===selected);
  const visibleRows=rowsKey===`${selected}:${offset}`?rows:[];
  return <section className="recording-workspace">
    <div className="recording-toolbar"><label>Collection<select value={selected} onChange={e=>{setSelected(e.target.value);setOffset(0);}} aria-label="Recording collection"><option value="">Choose a collection</option>{sessions.map(s=><option key={s.id} value={s.id}>{new Date(s.created*1000).toLocaleString()} · {s.captures} captures{s.active?' · recording':''}</option>)}</select></label>
      <button className="button secondary" onClick={()=>setRevision(r=>r+1)}>Refresh</button>
      <button className="button secondary" disabled={!selected} onClick={()=>void recordingRequest(`/${selected}/manifest`).then(value=>saveJson(value,`reddit-lens-${selected}-manifest.json`)).catch(e=>setError(e.message))}>Export manifest</button>
    </div>
    <p className="subtle">{active?`${active.captures} captures · ${(active.bytes/1024/1024).toFixed(1)} MB of image bytes · ${active.active?'Recording active':'Recording stopped'}`:'Start a recording session from extension Settings, then browse Reddit.'}</p>
    {root&&<p className="subtle">Saved locally at <code>{root}</code>. Originals and previews stay grouped. Nothing enters training automatically.</p>}
    <p className="subtle">Labels start unknown. A subreddit name or model score cannot verify origin. Add evidence before labeling an image real or AI.</p>
    {error&&<div className="error-message" role="alert">{error}</div>}
    {visibleRows.map(row=><ReviewRow key={row.id} row={row} session={selected} onSaved={()=>setRevision(r=>r+1)}/>)}
    {!visibleRows.length&&!error&&<div className="empty-benchmark"><p>{loading?'Loading recorded images…':'No recorded images on this page.'}</p></div>}
    {selected&&<div className="recording-toolbar"><button className="button secondary" disabled={!offset} onClick={()=>setOffset(o=>Math.max(0,o-50))}>Previous</button><span className="subtle">Page {offset/50+1}</span><button className="button secondary" disabled={visibleRows.length<50} onClick={()=>setOffset(o=>o+50)}>Next</button></div>}
  </section>;
}
