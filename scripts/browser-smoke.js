// Execute in the running development lab with agent-browser eval --stdin.
// Requires a selected image. This is an API/runtime check, not an accuracy benchmark.
(async () => {
  const image = document.querySelector('.dropzone img');
  if (!image) throw new Error('Select the diagnostic drawing first.');
  const blob = await (await fetch(image.currentSrc || image.src)).blob();
  const {analyze} = await import('/src/detection/client.ts');
  const {modelConfig} = await import('/src/detection/model.ts');
  const selected = modelConfig(JSON.parse(localStorage.getItem('lens-settings:v2') || '{}').model);
  const result = await analyze(blob, 'original', selected.id);
  if (!result.components || !Object.keys(result.components).length) throw new Error('The selected model components are missing.');
  const health = await (await fetch('/api/health')).json();
  if (health.availableModels?.[result.model]?.modelHash !== result.modelHash) throw new Error('Model profile mismatch.');
  return {kind: 'local-ensemble-runtime-smoke-only', health, result};
})();
