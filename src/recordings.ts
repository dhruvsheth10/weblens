import type {Capture} from './extension/scan';
export function backendUrl(path:string) {return location.protocol==='chrome-extension:'?`http://127.0.0.1:8000${path}`:path;}
export async function recordingRequest(path:string,options:RequestInit={}) {
  let response:Response;
  try {response=await fetch(backendUrl(`/api/recordings${path}`),{...options,signal:AbortSignal.timeout(30000)});}
  catch {throw new Error('Local recording backend unavailable. Start or restart npm run dev.');}
  const value=await response.json();
  if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:'Recording request failed.');
  return value;
}
export async function recordCapture(session:string,capture:Capture):Promise<{id:string;sessionId:string}> {
  const {preview,original,options,result}=capture;
  const form=new FormData();form.append('preview',preview.blob,'preview');if(original)form.append('original',original.blob,'original');
  form.append('metadata',JSON.stringify({schemaVersion:1,postUrl:options.context?.postUrl,subreddit:options.context?.subreddit,
    policy:options.source,threshold:options.threshold,preview:{url:preview.url,prediction:preview.prediction},
    original:original?{url:original.url,prediction:original.prediction}:null,selectedPrediction:result,
    capturedAt:new Date().toISOString(),fetch:result.input}));
  try {return await recordingRequest(`/${session}/captures`,{method:'POST',body:form});}
  catch(error) {
    // A capped/stopped session must not keep fetching record-only originals.
    if(location.protocol==='chrome-extension:')try {
      const status=await recordingRequest('');
      const active=status.sessions.some((value:{id:string;active:number})=>value.id===session&&value.active);
      if(!active)await chrome.runtime.sendMessage({target:'background',type:'RECORDING_STOPPED',session});
    }catch {/* Preserve the original actionable recording error when offline. */}
    throw error;
  }
}
