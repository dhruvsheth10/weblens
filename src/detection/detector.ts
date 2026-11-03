import {DEFAULT_MODEL_ID, MAX_BYTES, isRedditStudent, modelConfig, sigmoid, type Detection, type ModelId, type Variant} from './model';

export async function detect(blob: Blob, variant: Variant = 'original', modelId: ModelId = DEFAULT_MODEL_ID): Promise<Detection> {
  if (!blob.size || blob.size > MAX_BYTES) throw new Error('Choose an image smaller than 20 MB.');
  const endpoint = self.location.protocol === 'chrome-extension:'
    ? 'http://127.0.0.1:8000/api/detect' : new URL('/api/detect', self.location.href).href;
  const form = new FormData();
  form.append('file', blob, 'image'); form.append('variant', variant); form.append('model', modelId);
  let response: Response;
  try {response = await fetch(endpoint, {method: 'POST', body: form, signal: AbortSignal.timeout(600_000)});}
  catch {throw new Error('The local detector backend is unavailable. Start it with npm run dev, then retry.');}
  const result = await response.json().catch(() => {throw new Error('The local backend did not return a result. Check that npm run dev is running.');});
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Local inference failed. Check the backend terminal.');
  const expected = modelConfig(modelId);
  if (result.model !== expected.id || result.modelHash !== expected.sha256 || result.pipeline !== expected.pipeline) {
    throw new Error('The page and local models have different versions. Reload the page after setup.');
  }
  const requiredComponents = isRedditStudent(modelId) ? ['reddit_student']
    : modelId === 'mirror-h-v1' ? ['mirror']
    : modelId === 'mirror-community-v1' ? ['mirror','community_forensics','ferretnet_b']
    : modelId === 'community-ferret-preview-v1' ? ['community_forensics','ferretnet_b']
    : modelId === 'community-preview-v1' ? ['community_forensics']
    : ['ferretnet_b', 'opensynthid'];
  if (modelId.startsWith('community-') && result.uncertainThreshold !== expected.uncertainThreshold) {
    throw new Error('The semantic detector decision thresholds differ from this page. Reload after setup.');
  }
  if (modelId === 'community-ferret-preview-v1' && (!Number.isFinite(result.semanticScore) || result.semanticScore !== result.components?.community_forensics?.score)) {
    throw new Error('The combined detector returned an invalid semantic score.');
  }
  if (!Number.isFinite(result.score) || result.score < 0 || result.score > 1 ||
      !Number.isFinite(result.logit) || Math.abs(sigmoid(result.logit) - result.score) > 1e-7 ||
      !result.components || requiredComponents.some(key => !result.components[key]) || result.variant !== variant ||
      !['cpu', 'mps', 'cuda'].includes(result.engine)) throw new Error('The backend returned an invalid ensemble result.');
  return result as Detection;
}
