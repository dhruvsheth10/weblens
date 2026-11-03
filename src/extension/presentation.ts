import {decisionBand,type Detection,type ModelConfig} from '../detection/model';
import type {ImageSource} from './images';

export const SOURCE_HELP:Record<ImageSource,string>={
  'preview-first':'Checks the image Reddit displays first. If its signal is uncertain, checks the original too. Recommended for faster browsing with fewer original downloads.',
  'prefer-original':'Tries the full original image for every scan. Uses the displayed preview if that fails. More thorough, with larger downloads.',
  'displayed':'Checks only the image displayed in the feed. Uses the least data, but Reddit compression can hide AI signals.',
};
export type ImageMark={state:'scanning'|'ai'|'uncertain'|'unmarked'|'error';label:string;score?:string;detail:string;title:string;busy?:boolean};
export function detectedMark(result:Detection,model:ModelConfig,threshold:number,pending=false):ImageMark {
  const band=decisionBand(result.score,threshold,model.uncertainThreshold,result.semanticScore);
  return {state:band,label:band==='ai'?'AI signal detected':band==='uncertain'?'Uncertain AI signal':'Scanned · no strong AI signal',
    score:`${(result.score*100).toFixed(1)}%`,detail:pending?'Preview result · scan finishing…':result.recordingError?'Scanned · recording failed':'Raw model score',busy:pending,
    title:`Raw detector score ${result.score.toFixed(6)}. This is not a probability or proof of AI origin. A low score does not establish that an image is real.\n${result.model} · ${result.engine}${result.input?`\nAnalyzed ${result.input.inputSource}: ${result.input.sourceUrl}\nOriginal: ${result.input.originalStatus}${result.input.fallbackReason?`\n${result.input.fallbackReason}`:''}`:''}${result.recordingError?`\nRecording: ${result.recordingError}`:''}`};
}
export type PageStatus={mode:'starting'|'paused'|'scope'|'checking'|'scanning'|'ready'|'error';label:string;detail:string;scanned:number;ai:number;uncertain:number;errors:number;queued:number;recording:boolean;model:string};
export function summarizeResults(values:Iterable<{state:ImageMark['state']}>) {
  let scanned=0,ai=0,uncertain=0,errors=0;
  for(const value of values){if(value.state==='scanning')continue;if(value.state==='error')errors++;else{scanned++;if(value.state==='ai')ai++;else if(value.state==='uncertain')uncertain++;}}
  return {scanned,ai,uncertain,errors};
}
