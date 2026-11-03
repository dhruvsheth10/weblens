"""Install explicit adapters only after the running baseline benchmark finishes."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'benchmark_mcp/core.py';s=p.read_text()
if "if model.startswith('reddit-'):" not in s:
 s=s.replace("    if model.startswith('mirror-'):","    if model.startswith('reddit-'):\n        files+=['backend/reddit_research.py','backend/reddit_research_profile.py','backend/mirror.py','backend/mirror_profile.py','backend/vendor/mirror/upstream.py','backend/community.py','backend/semantic_profile.py','backend/ferretnet.py','backend/models/community-profile.json','backend/models/community-ferret-profile.json']\n    elif model.startswith('mirror-'):")
 s=s.replace("    if profile.get('kind') in {'mirror','mirror-hybrid'}:","    if profile.get('kind') in {'reddit-fusion','reddit-student'}:\n        from backend.reddit_research_profile import validate_profile as validate_reddit\n        return validate_reddit(profile)\n    if profile.get('kind') in {'mirror','mirror-hybrid'}:")
 p.write_text(s)
p=root/'benchmark_mcp/worker.py';s=p.read_text()
if "model['profile']['kind'] in {'reddit-fusion','reddit-student'}" not in s:
 s=s.replace("    if model['kind']=='profile' and model['profile']['kind'] in {'mirror','mirror-hybrid'}:","    if model['kind']=='profile' and model['profile']['kind'] in {'reddit-fusion','reddit-student'}:\n        from backend.reddit_research import RedditFusionEngine,RedditStudentEngine\n        engine=(RedditFusionEngine if model['profile']['kind']=='reddit-fusion' else RedditStudentEngine)(profile=model['profile'])\n        cutoff=model['profile']['threshold']\n    elif model['kind']=='profile' and model['profile']['kind'] in {'mirror','mirror-hybrid'}:")
 p.write_text(s)
print('Explicit Reddit research adapters enabled')
