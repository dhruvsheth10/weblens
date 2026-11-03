import student from './reddit-student-profile.json';
import sherlock from './sherlock-profile.json';
import sherlock2 from './sherlock-2-profile.json';
import mirror from './mirror-profile.json';
import mirrorCommunity from './mirror-community-profile.json';
import hybrid from './hybrid-profile.json';
import community from './community-profile.json';
import communityFerret from './community-ferret-profile.json';
export type ModelId = 'reddit-sherlock-2' | 'reddit-sherlock-1.0' | 'reddit-student-last2-v1' | 'mirror-h-v1' | 'mirror-community-v1' | 'ferretnet-opensynthid-v1' | 'community-preview-v1' | 'community-ferret-preview-v1';
export interface ModelConfig {
  id: ModelId; name: string; shortName: string; license: string; repository: string;
  sha256: string; pipeline: string; threshold: number; description: string;
  uncertainThreshold?: number;
}
export const MODELS: Record<ModelId, ModelConfig> = {
  'community-ferret-preview-v1': {
    id:'community-ferret-preview-v1',name:'Community Preview + FerretNet',shortName:'Community + Ferret',
    license:'MIT + Apache 2.0',repository:'https://huggingface.co/OwensLab/commfor-model-224',
    sha256:communityFerret.modelHash,pipeline:communityFerret.pipeline,threshold:communityFerret.threshold,uncertainThreshold:communityFerret.uncertainThreshold,
    description:'Semantic preview detector + native pixel signal · shared image decode · experimental',
  },
  'reddit-sherlock-2': {
    id:'reddit-sherlock-2',name:'Sherlock 2 · experimental',shortName:'Sherlock 2',
    license:'Research model · data terms apply',repository:'https://huggingface.co/OwensLab/commfor-model-224',
    sha256:sherlock2.modelHash,pipeline:sherlock2.pipeline,threshold:sherlock2.threshold,
    description:'Stricter Reddit preview student · fewer false flags than Sherlock 1.0, and it catches fewer AI images · displayed images recommended',
  },
  'reddit-sherlock-1.0': {
    id:'reddit-sherlock-1.0',name:'Sherlock 1.0 · experimental',shortName:'Sherlock 1.0',
    license:'Research model · data terms apply',repository:'https://huggingface.co/OwensLab/commfor-model-224',
    sha256:sherlock.modelHash,pipeline:sherlock.pipeline,threshold:sherlock.threshold,
    description:'Fully fine-tuned Reddit preview student · ~0.5% false flags on held-out Reddit · displayed images recommended',
  },
  'reddit-student-last2-v1': {
    id:'reddit-student-last2-v1',name:'Reddit Preview Student · experimental',shortName:'Reddit Student',
    license:'Research model · data terms apply',repository:'https://huggingface.co/OwensLab/commfor-model-224',
    sha256:student.modelHash,pipeline:student.pipeline,threshold:student.threshold,
    description:'Small preview detector · conservative cutoff · experimental · displayed images recommended',
  },
  'community-preview-v1': {
    id:'community-preview-v1',name:'Community Forensics · preview head',shortName:'Community Preview',
    license:'MIT',repository:'https://huggingface.co/OwensLab/commfor-model-224',
    sha256:community.modelHash,pipeline:community.pipeline,threshold:community.threshold,uncertainThreshold:community.uncertainThreshold,
    description:'Small semantic detector · JPEG / WebP training · experimental local head',
  },
  'ferretnet-opensynthid-v1': {
    id: 'ferretnet-opensynthid-v1', name: 'FerretNet + OpenSynthID', shortName: 'Ferret + OpenSynthID',
    license: 'Apache 2.0', repository: 'https://huggingface.co/fyxme/opensynthid-detect-0.1',
    sha256: hybrid.modelHash, pipeline: hybrid.pipeline, threshold: hybrid.threshold,
    description: 'Two local detectors · additional Gemini signals · experimental, not verified provenance',
  },
  'mirror-h-v1': {
    id:'mirror-h-v1',name:'MIRROR H+ · experimental',shortName:'MIRROR H+',
    license:'Research weights · upstream terms',repository:'https://github.com/handsome-rich/MIRROR',
    sha256:mirror.modelHash,pipeline:mirror.pipeline,threshold:mirror.threshold,
    description:'Large local detector · 3.45 GB checkpoint · loads on first scan',
  },
  'mirror-community-v1': {
    id:'mirror-community-v1',name:'MIRROR + Community + FerretNet',shortName:'MIRROR + Community + Ferret',
    license:'Research weights · upstream terms',repository:'https://github.com/handsome-rich/MIRROR',
    sha256:mirrorCommunity.modelHash,pipeline:mirrorCommunity.pipeline,threshold:mirrorCommunity.threshold,
    description:'Experimental combination · flags if either detector crosses its cutoff · slower',
  },
};
export const DEFAULT_MODEL_ID: ModelId = 'reddit-sherlock-2';
export const MODEL = MODELS[DEFAULT_MODEL_ID];
export const DEFAULT_THRESHOLD = MODEL.threshold;
export const PIPELINE = MODEL.pipeline;
export function isRedditStudent(id: string) {
  return id === 'reddit-sherlock-2' || id === 'reddit-sherlock-1.0' || id === 'reddit-student-last2-v1';
}
export function modelConfig(id: string): ModelConfig {
  return Object.hasOwn(MODELS,id) ? MODELS[id as ModelId] : MODEL;
}
export const MAX_BYTES = 20 * 1024 * 1024;
export type Variant = 'original' | 'jpeg75' | 'resize512' | 'reddit-webp640' | 'reddit-webp960';
export function variantLabel(variant: Variant) {
  return {original:'Original image',jpeg75:'JPEG · quality 75',resize512:'Resize · max 512 px',
    'reddit-webp640':'WebP · 640 px width','reddit-webp960':'WebP · 960 px width'}[variant];
}
export interface Detection {
  model: string; modelHash: string; pipeline: string; score: number; logit: number;
  width: number; height: number; elapsedMs: number; engine: 'cpu' | 'mps' | 'cuda';
  variant: Variant;
  watermarkVerified?: boolean;
  uncertainThreshold?: number;
  semanticScore?: number;
  input?: {sourceUrl: string; displayedUrl: string; inputSource: 'original' | 'displayed'; contentType: string; downloadBytes: number; downloadMs: number; fallbackReason?: string;
    policy?:string; originalUrl?:string;originalStatus?:string;escalated?:boolean;previewScore?:number;previewSemanticScore?:number;
    attempts?:{url:string;role:string;ok:boolean;status?:number;bytes?:number;ms:number;error?:string}[];totalMs?:number};
  recording?:{id:string;sessionId:string};
  recordingError?:string;
  components?: Record<string, {logit: number; score: number; calibratedScore?: number; weight: number; [key: string]: unknown}>;
}
export type Label = 'real' | 'ai';
export interface BenchmarkRow extends Detection {
  file: string; label: Label; group: string; split: 'calibration' | 'validation';
  sourceHash: string;
}
export function sigmoid(logit: number) {
  if (!Number.isFinite(logit)) throw new Error('The detector returned a non-finite logit.');
  return logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
}
export function decisionBand(score: number, threshold = DEFAULT_THRESHOLD, uncertainThreshold?: number, uncertainScore = score) {
  return score >= threshold ? 'ai' : uncertainThreshold !== undefined && uncertainScore >= Math.min(uncertainThreshold,threshold) ? 'uncertain' : 'unmarked';
}
export function verdict(score: number, threshold = DEFAULT_THRESHOLD, uncertainThreshold?: number, uncertainScore = score) {
  const band=decisionBand(score,threshold,uncertainThreshold,uncertainScore);
  return band==='ai' ? 'Strong AI signal' : band==='uncertain' ? 'Uncertain AI signal' : 'No strong AI signal';
}
export function isRedditImageUrl(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password && !url.port &&
      ['i.redd.it', 'preview.redd.it', 'external-preview.redd.it'].includes(url.hostname);
  } catch { return false; }
}
