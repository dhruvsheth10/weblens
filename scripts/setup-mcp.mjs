import {spawnSync} from 'node:child_process';
import {existsSync} from 'node:fs';

function run(command, args) {
  const result = spawnSync(command, args, {stdio: 'inherit'});
  if (result.error || result.status !== 0) throw result.error ?? new Error(`${command} failed (${result.status})`);
}
if (!existsSync('backend/.venv/bin/python')) throw new Error('Run npm run setup first.');
if (!existsSync('benchmark_mcp/.venv/bin/python')) run('backend/.venv/bin/python', ['-m', 'venv', 'benchmark_mcp/.venv']);
run('benchmark_mcp/.venv/bin/python', ['-m', 'pip', 'install', '--no-user', '-r', 'benchmark_mcp/requirements.txt']);
console.log('Local benchmark MCP ready. See docs/BENCHMARK-MCP.md for configuration and usage.');
