import {spawn} from 'node:child_process';
import {existsSync,readFileSync} from 'node:fs';
import {setTimeout as delay} from 'node:timers/promises';

if (!existsSync('backend/models/profile.json')) throw new Error('Run npm run setup before starting FiSeR + DEAR-r.');
if (!existsSync('backend/models/ferretnet_b.pth')) throw new Error('Run npm run setup before starting FerretNet-B.');
const bootstrapModel = {id: 'ferretnet-b-v1', hash: 'fe755d78370bb6547070329553572405b4ecebd23382c9a6cbb11c4ab85a82c2'};
const student=JSON.parse(readFileSync('backend/models/reddit-student-profile.json','utf8'));
const mirror=JSON.parse(readFileSync('backend/models/mirror-profile.json','utf8'));
const prototype=JSON.parse(readFileSync('backend/models/sherlock-2-profile.json','utf8'));
const pickerProfiles=['community','hybrid','mirror-community'].map(name=>JSON.parse(readFileSync(`backend/models/${name}-profile.json`,'utf8')));
function compatible(health) {
  const selected=health.availableModels?.[prototype.model];
  return health.protocolVersion===2 && health.model===bootstrapModel.id && health.modelHash===bootstrapModel.hash &&
    selected?.status==='ready' && selected.modelHash===prototype.modelHash && health.availableModels?.[mirror.model]?.modelHash===mirror.modelHash && health.availableModels?.[student.model]?.modelHash===student.modelHash && pickerProfiles.every(profile=>health.availableModels?.[profile.model]?.modelHash===profile.modelHash);
}
const children = [];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) child.kill('SIGTERM');
  setTimeout(() => process.exit(code), 500).unref();
}
function launch(command, args) {
  const child = spawn(command, args, {stdio: 'inherit', env: process.env});
  children.push(child);
  child.on('error', error => {console.error(error.message); stop(1);});
  child.on('exit', code => {if (!stopping) stop(code ?? 1);});
  return child;
}
process.on('SIGINT', () => stop()); process.on('SIGTERM', () => stop());
let ready = false;
let incompatible = false;
try {
  const response = await fetch('http://127.0.0.1:8000/api/health');
  if (response.ok) {const health = await response.json(); ready = compatible(health); incompatible = !ready;}
} catch {}
if (incompatible) throw new Error('An older backend is using port 8000. Stop it, then run npm run dev again.');
if (!ready) launch('backend/.venv/bin/python', ['-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '8000']);
for (let tries = 0; tries < 120 && !ready && !stopping; tries++) {
  await delay(500);
  try {const response = await fetch('http://127.0.0.1:8000/api/health'); const health = response.ok ? await response.json() : {}; ready = compatible(health);} catch {}
}
if (!ready) {console.error('The local models could not start. Check the backend output.'); stop(1);}
else launch('node', ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173', '--strictPort']);
