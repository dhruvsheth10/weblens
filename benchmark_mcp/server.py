"""STDIO MCP; stdout is reserved for protocol traffic."""
from __future__ import annotations
import argparse
import os
import sys
from functools import wraps
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from benchmark_mcp.core import BenchmarkManager
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

def client_errors(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        try:
            return function(*args,**kwargs)
        except (ValueError,OSError,KeyError) as error:
            raise ToolError(str(error)) from error
    return wrapped

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',default=os.environ.get('LENS_BENCHMARK_CALIBRATION_DIR'))
    parser.add_argument('--manifest',default=os.environ.get('LENS_BENCHMARK_SPLIT_MANIFEST'))
    parser.add_argument('--manifest-sha',default=os.environ.get('LENS_BENCHMARK_SPLIT_MANIFEST_SHA'))
    parser.add_argument('--output',default=str(PROJECT/'artifacts/mcp-benchmarks'))
    parser.add_argument('--device',choices=['cpu','mps','cuda'])
    parser.add_argument('--external-config',default=os.environ.get('LENS_BENCHMARK_EXTERNAL_CONFIG'))
    args=parser.parse_args()
    if not args.calibration:
        parser.error('An explicit calibration directory is required; no dataset discovery is performed.')
    manager=BenchmarkManager(PROJECT,args.calibration,args.output,args.manifest,device=args.device,manifest_sha=args.manifest_sha,external_config=args.external_config)
    @asynccontextmanager
    async def lifespan(server):
        try:
            yield manager
        finally:
            manager.close()
    server=MCPServer('redditscanner-benchmarks',version='1.0.0',lifespan=lifespan,
        instructions='Local AI-image benchmarks. Inspect benchmark_info first. Validation and the reserved calibration audit are disabled. Start jobs, poll get_run, then compare completed runs. Numerical FerretNet profiles are immutable and checksummed.')
    @server.tool(structured_output=True)
    @client_errors
    def benchmark_info() -> dict[str, Any]:
        """Show configured dataset scope, partitions and output directory."""
        return manager.info()
    @server.tool(structured_output=True)
    @client_errors
    def list_models() -> dict[str, Any]:
        """List builtins and registered immutable model/profile versions."""
        return {'models':manager.models()}
    @server.tool(structured_output=True)
    @client_errors
    def register_model(profile_path: str) -> dict[str, Any]:
        """Validate and snapshot a numerical FerretNet profile JSON inside the project."""
        return manager.register(profile_path)
    @server.tool(structured_output=True)
    @client_errors
    def start_benchmark(version: str, partition: str | None = None, threshold: float | None = None, variant: str = 'original', dataset: str = 'local') -> dict[str, Any]:
        """Start one background benchmark. Use builtin ID or registered SHA version; poll get_run."""
        return manager.start(version,partition,threshold,variant,dataset)
    @server.tool(structured_output=True)
    @client_errors
    def get_run(run_id: str) -> dict[str, Any]:
        """Return progress, failures and final precision/recall/F1/timing with artifact paths."""
        return manager.get(run_id)
    @server.tool(structured_output=True)
    @client_errors
    def list_runs(limit: int = 20) -> dict[str, Any]:
        """List recent persisted runs, including jobs from earlier server sessions."""
        return {'runs':manager.runs(limit)}
    @server.tool(structured_output=True)
    @client_errors
    def cancel_run(run_id: str) -> dict[str, Any]:
        """Cancel a running worker and retain partial scores, status and logs."""
        return manager.cancel(run_id)
    @server.tool(structured_output=True)
    @client_errors
    def compare_runs(run_ids: list[str]) -> dict[str, Any]:
        """Compare two to ten complete runs on exactly matching images, scope, variant and device."""
        return manager.compare(run_ids)
    server.run()

if __name__=='__main__':
    main()
