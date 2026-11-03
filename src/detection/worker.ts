import { detect } from './detector';
import type { ModelId, Variant } from './model';
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<{id: string; blob: Blob; variant: Variant; model: ModelId}>) => {
  const { id, blob, variant, model } = event.data;
  queue = queue.then(async () => {
    try { self.postMessage({id, result: await detect(blob, variant, model)}); }
    catch (error) { self.postMessage({id, error: error instanceof Error ? error.message : String(error)}); }
  });
};
