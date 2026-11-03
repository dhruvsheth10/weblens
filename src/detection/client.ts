import {DEFAULT_MODEL_ID, type Detection, type ModelId, type Variant} from './model';
let worker: Worker | undefined;
const pending = new Map<string, {resolve: (d: Detection) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout>}>();
export function analyze(blob: Blob, variant: Variant = 'original', model: ModelId = DEFAULT_MODEL_ID): Promise<Detection> {
  if (!worker) {
    worker = location.protocol === 'chrome-extension:'
      ? new Worker(new URL('/inference-worker.js', location.href), {type: 'module'})
      : new Worker(new URL('./worker.ts', import.meta.url), {type: 'module'});
    worker.onmessage = ({data}) => {
      const task = pending.get(data.id); if (!task) return;
      clearTimeout(task.timer); pending.delete(data.id);
      if (data.error) task.reject(new Error(data.error)); else task.resolve(data.result);
    };
    worker.onerror = () => resetAnalyzer('The inference worker stopped. Retry the image.');
  }
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    const timer = setTimeout(() => resetAnalyzer('Local analysis timed out. Try a smaller image and check the backend terminal.'), 600_000);
    pending.set(id, {resolve, reject, timer}); worker!.postMessage({id, blob, variant, model});
  });
}
export function resetAnalyzer(reason = 'Analysis cancelled.') {
  worker?.terminate(); worker = undefined;
  for (const task of pending.values()) {clearTimeout(task.timer); task.reject(new Error(reason));}
  pending.clear();
}
