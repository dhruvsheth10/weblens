"""Score explicitly named images with a pinned local research candidate."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.reddit_research import RedditFusionEngine,RedditStudentEngine
from backend.community import CommunityEngine

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('images',type=Path,nargs='+');args=parser.parse_args()
 profile_path=args.profile.resolve()
 if not profile_path.is_relative_to(ROOT) or profile_path.suffix!='.json' or profile_path.stat().st_size>1_000_000:parser.error('Use a project-local numerical profile JSON')
 manifest=json.loads((ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json').read_text());reserved={str(Path('/Users/dhruvsheth/Downloads/benchmark')/r['file']) for r in manifest['rows'] if r['experimentSplit']=='audit'}
 frozen=json.loads((ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-splits.json').read_text())['rows'];reserved.update(str(ROOT/'datasets/reddit/data'/r[k]['path']) for r in frozen if r['split'] in {'holdout','quarantine'} for k in ['preview','original'] if (r.get(k) or {}).get('path'))
 checked=[]
 for candidate in args.images:
  absolute=candidate.absolute()
  if 'validation' in [x.lower() for x in absolute.parts] or str(absolute) in reserved:parser.error('Reserved validation/audit/holdout image access is disabled')
  if candidate.is_symlink():parser.error('Symlinked images are refused')
  path=candidate.resolve()
  if str(path) in reserved:parser.error('Reserved image access is disabled')
  if not path.is_file():parser.error('Pass explicit image files, not directories')
  checked.append(path)
 p=json.loads(profile_path.read_text());kind=p.get('kind');classes={'reddit-fusion':RedditFusionEngine,'reddit-student':RedditStudentEngine,'semantic-head':CommunityEngine}
 if kind not in classes:parser.error('Unsupported research candidate kind')
 engine=classes[kind](profile=p)
 for path in checked:
  print(json.dumps({'file':str(path),**engine.predict(path.read_bytes())},allow_nan=False),flush=True)
if __name__=='__main__':main()
