import {spawnSync} from 'node:child_process';
import {existsSync} from 'node:fs';

function run(command, args) {
  const result = spawnSync(command, args, {stdio: 'inherit', env: process.env});
  if (result.error || result.status !== 0) throw result.error ?? new Error(`${command} exited ${result.status}`);
}
if (!existsSync('backend/.venv/bin/python')) {
  const configured = process.env.LENS_PYTHON;
  const python = configured ?? ['python3.12', 'python3.13', 'python3.14', 'python3'].find(name => spawnSync(name, ['--version']).status === 0);
  if (!python) throw new Error('Install Python 3.12 or newer, or set LENS_PYTHON.');
  run(python, ['-m', 'venv', 'backend/.venv']);
}
run('backend/.venv/bin/python', ['-m', 'pip', '--isolated', 'install', '-r', 'backend/requirements.txt']);
run('backend/.venv/bin/python', ['-u', '-m', 'backend.train_head']);
console.log('FerretNet-B, Community Preview, FerretNet + OpenSynthID, and FiSeR + DEAR-r are ready. Run npm run dev.');
