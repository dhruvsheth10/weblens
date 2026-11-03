import { DEFAULT_MODEL_ID, MODELS, isRedditStudent, modelConfig, type ModelId } from '../detection/model';
import {type ImageSource} from './images';
export interface Settings { settingsVersion: 4; enabled: boolean; feed: boolean; threshold: number; model: ModelId; imageSource: ImageSource; recordingSession?:string }
export const DEFAULT_SETTINGS: Settings = {
  settingsVersion: 4, enabled: false, feed: false,
  threshold: MODELS[DEFAULT_MODEL_ID].threshold, model: DEFAULT_MODEL_ID,
  imageSource: 'displayed',
};
export function sourceForModelSwitch(source: ImageSource, model: ModelId): ImageSource {
  return isRedditStudent(model) ? 'displayed'
    : source === 'preview-first' && modelConfig(model).uncertainThreshold === undefined ? 'prefer-original' : source;
}
export async function getSettings(): Promise<Settings> {
  const stored = (await chrome.storage.local.get('settings')).settings ?? {};
  const current = stored.settingsVersion === 4;
  const model = current && Object.hasOwn(MODELS,stored.model) ? stored.model as ModelId : DEFAULT_MODEL_ID;
  return {settingsVersion: 4, enabled: stored.enabled === true, feed: stored.feed === true, model,
    imageSource: !current ? DEFAULT_SETTINGS.imageSource : stored.imageSource === 'displayed' ? 'displayed' : stored.imageSource === 'prefer-original' ? 'prefer-original' : isRedditStudent(model)?'displayed':modelConfig(model).uncertainThreshold!==undefined?'preview-first':'prefer-original',
    // Pair collection is disabled, including sessions saved by older releases.
    recordingSession:undefined,
    threshold: current && stored.model === model && Number.isFinite(stored.threshold) && stored.threshold >= 0 && stored.threshold <= 1
      ? stored.threshold : modelConfig(model).threshold};
}
