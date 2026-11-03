import {build} from 'esbuild';
import {mkdir, cp, writeFile, readFile, stat, rm} from 'node:fs/promises';
const out = 'extension-dist';
// Stop before replacing the packaged extension if its model identities drifted
// from the local backend. No model weights are copied into the extension.
for(const name of ['community-ferret','community','hybrid','mirror','mirror-community','reddit-student','sherlock','sherlock-2']) {
  const client=JSON.parse(await readFile(`src/detection/${name}-profile.json`,'utf8'));
  const backend=JSON.parse(await readFile(`backend/models/${name}-profile.json`,'utf8'));
  for(const key of ['model','modelHash','pipeline','threshold','uncertainThreshold']) {
    if(client[key]!==backend[key])throw new Error(`${name}: extension/backend ${key} mismatch. Synchronize the pinned profiles before building.`);
  }
}
await rm(out, {recursive: true, force: true});
await mkdir(out, {recursive: true});
await cp('dist', out, {recursive: true});
await build({entryPoints: {'background': 'src/extension/background.ts', 'offscreen': 'src/extension/offscreen.ts', 'inference-worker': 'src/detection/worker.ts', 'popup': 'src/extension/popup.ts'}, outdir: out, bundle: true, format: 'esm', target: 'chrome120', minify: true, define: {'process.env.NODE_ENV': '"production"'}});
await build({entryPoints: ['src/extension/content.ts'], outfile: `${out}/content.js`, bundle: true, format: 'iife', target: 'chrome120', minify: true});
await cp('extension', out, {recursive: true, filter: source => !source.startsWith('extension/fonts')});
const manifest = JSON.parse(await readFile('extension/manifest.json', 'utf8'));
for (const file of [manifest.background.service_worker, manifest.action.default_popup, manifest.options_page, 'offscreen.html', 'inference-worker.js', 'content.js', 'theme.css', 'fonts/google-sans-latin.woff2', 'fonts/OFL.txt', 'fonts/SOURCE.json']) await stat(`${out}/${file}`);
const sherlock2 = JSON.parse(await readFile('src/detection/sherlock-2-profile.json','utf8'));
const sherlock = JSON.parse(await readFile('src/detection/sherlock-profile.json','utf8'));
const student = JSON.parse(await readFile('src/detection/reddit-student-profile.json','utf8'));
const mirror = JSON.parse(await readFile('src/detection/mirror-profile.json', 'utf8'));
const mirrorCommunity = JSON.parse(await readFile('src/detection/mirror-community-profile.json', 'utf8'));
const hybrid = JSON.parse(await readFile('src/detection/hybrid-profile.json','utf8'));
const communityFerret = JSON.parse(await readFile('src/detection/community-ferret-profile.json','utf8'));
const community = JSON.parse(await readFile('src/detection/community-profile.json','utf8'));
await writeFile(`${out}/build-info.json`, JSON.stringify({
  builtAt: new Date().toISOString(), defaultModel: sherlock2.model,
  modelHash: sherlock2.modelHash,
  availableModels: [{id:sherlock2.model,modelHash:sherlock2.modelHash},{id:sherlock.model,modelHash:sherlock.modelHash},{id:student.model,modelHash:student.modelHash},{id:'mirror-h-v1',modelHash:mirror.modelHash},{id:'mirror-community-v1',modelHash:mirrorCommunity.modelHash}, {id:'ferretnet-opensynthid-v1',modelHash:hybrid.modelHash},
    {id:'community-preview-v1',modelHash:community.modelHash},{id:'community-ferret-preview-v1',modelHash:communityFerret.modelHash}],
  backend: 'http://127.0.0.1:8000',
}, null, 2));
console.log('Load extension-dist in chrome://extensions → Load unpacked.');
