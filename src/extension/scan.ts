import {decisionBand,modelConfig,type Detection,type ModelId} from '../detection/model';
import {fetchImage,originalCandidate,ImageFetchError,type ImageSource,type PostContext} from './images';

export type FetchAttempt={url:string; role:'displayed'|'original'; ok:boolean; status?:number; bytes?:number; ms:number; error?:string};
export type ScanOptions={url:string; model:ModelId; source:ImageSource; threshold:number; context?:PostContext; record?:boolean};
export type Capture={options:ScanOptions; preview:{blob:Blob;url:string;prediction:Detection}; original?:{blob:Blob;url:string;prediction:Detection}; result:Detection};
export type ScanProgress={phase:'fetching'|'analyzing'|'recording';role:'displayed'|'original'};
export type ScanDeps={fetcher?:typeof fetch; analyze:(blob:Blob,model:ModelId)=>Promise<Detection>; onPreview?:(result:Detection)=>void; onProgress?:(progress:ScanProgress)=>void; record?:(capture:Capture)=>Promise<{id:string;sessionId:string}>};

export async function scanImage(options:ScanOptions,deps:ScanDeps):Promise<Detection> {
  const started=performance.now();const model=modelConfig(options.model);
  const originalUrl=originalCandidate(options.url,options.context?.candidates);
  const attempts:FetchAttempt[]=[];
  const get=async(url:string,role:FetchAttempt['role'])=>{
    deps.onProgress?.({phase:'fetching',role});
    const at=performance.now();
    try {const value=await fetchImage(url,deps.fetcher);attempts.push({url,role,ok:true,status:value.status,bytes:value.blob.size,ms:Math.round(performance.now()-at)});return value;}
    catch(error){attempts.push({url,role,ok:false,ms:Math.round(performance.now()-at),...(error instanceof ImageFetchError?{status:error.status}:{}),error:error instanceof Error?error.message:String(error)});throw error;}
  };
  let preview:Capture['preview']|undefined,original:Capture['original']|undefined;
  let fallbackReason:string|undefined;
  let recordingError:string|undefined;
  const decode=async(value:{blob:Blob;url:string},role:ScanProgress['role'])=>{deps.onProgress?.({phase:'analyzing',role});return {...value,prediction:await deps.analyze(value.blob,options.model)};};
  let result:Detection;
  let escalated=false;
  const source=options.source==='preview-first' && model.uncertainThreshold===undefined?'prefer-original':options.source;
  if(source==='prefer-original' && originalUrl && originalUrl!==options.url) {
    try {original=await decode(await get(originalUrl,'original'),'original');result=original.prediction;}
    catch(error){fallbackReason=String(error instanceof Error?error.message:error);preview=await decode(await get(options.url,'displayed'),'displayed');result=preview.prediction;}
  } else {
    preview=await decode(await get(options.url,'displayed'),'displayed');result=preview.prediction;
    deps.onPreview?.(result);
    escalated=source==='preview-first' && decisionBand(result.score,options.threshold,model.uncertainThreshold,result.semanticScore)==='uncertain';
    if(escalated && originalUrl && originalUrl!==options.url) {
      try {original=await decode(await get(originalUrl,'original'),'original');result=original.prediction;}
      catch(error){fallbackReason=String(error instanceof Error?error.message:error);}
    }
  }
  // Recording independently collects both versions; it cannot change the badge decision.
  if(options.record) {
    if(!preview)try {preview=await decode(await get(options.url,'displayed'),'displayed');}catch(error){recordingError=`Recording preview unavailable: ${error instanceof Error?error.message:String(error)}`;}
    if(!original && originalUrl===options.url && preview)original=preview;
    if(!original && originalUrl && originalUrl!==options.url && !attempts.some(a=>a.role==='original')) {
      try {original=await decode(await get(originalUrl,'original'),'original');}catch(error){fallbackReason ??=String(error instanceof Error?error.message:error);}
    }
  }
  const chosen=original && result===original.prediction?original:preview;
  if(!chosen)throw new Error('No analyzed image available.');
  const unresolved=!originalUrl?'No matching original URL in post metadata or supported preview filename.':undefined;
  result={...result,input:{sourceUrl:chosen.url,displayedUrl:options.url,inputSource:new URL(chosen.url).hostname==='i.redd.it'?'original':'displayed',
    contentType:chosen.blob.type,downloadBytes:attempts.reduce((n,a)=>n+(a.bytes??0),0),downloadMs:attempts.reduce((n,a)=>n+a.ms,0),
    fallbackReason:fallbackReason ?? ((escalated || source==='prefer-original')?unresolved:undefined),
    policy:source,originalUrl,originalStatus:original?'fetched':attempts.some(a=>a.role==='original')?'failed':!originalUrl?'unresolved':'not-needed',
    escalated,previewScore:preview?.prediction.score,previewSemanticScore:preview?.prediction.semanticScore,attempts,totalMs:Math.round(performance.now()-started)}};
  if(originalUrl===options.url)result.input!.originalStatus='same-as-displayed';
  if(options.record && !deps.record)recordingError ??='Local recording is unavailable. Retry after reconnecting the extension.';
  if(recordingError)result={...result,recordingError};
  if(options.record && preview && deps.record) {
    deps.onProgress?.({phase:'recording',role:'original'});
    try {result={...result,recording:await deps.record({options,preview,original,result})};}
    catch(error){result={...result,recordingError:error instanceof Error?error.message:String(error)};}
  }
  return result;
}
