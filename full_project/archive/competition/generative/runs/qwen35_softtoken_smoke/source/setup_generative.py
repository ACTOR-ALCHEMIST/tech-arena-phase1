"""Create the isolated local LLM environment and fetch a pinned model revision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/generative_smoke.json')
    parser.add_argument('--download-only',action='store_true');args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text());python=ROOT/'.venv-generative/bin/python'
    out=ROOT/'competition/generative';out.mkdir(parents=True,exist_ok=True)
    if not args.download_only:
        if not python.exists():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(ROOT/'.venv-generative')],check=True)
        subprocess.run([str(python),'-m','pip','install','--timeout','120','-r',str(ROOT/'requirements-generative.txt')],check=True)
    # This subprocess uses the isolated Hugging Face dependency version.
    script="""
import json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
c=json.loads(Path(__import__('sys').argv[1]).read_text())
p=Path(snapshot_download(c['model_id'],revision=c['revision'],cache_dir='.tools/hf-generative',
    allow_patterns=['*.safetensors','*.json','*.txt','*.jinja'],max_workers=2)).resolve()
source={'repo_id':c['model_id'],'revision':c['revision'],'local_path':str(p),
        'weight_sha256':{f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in p.glob('*.safetensors')}}
Path('competition/generative/model_source.json').write_text(json.dumps(source,indent=2))
print(json.dumps(source,indent=2))
"""
    env=dict(os.environ,HF_HUB_DOWNLOAD_TIMEOUT='120')
    subprocess.run([str(python),'-c',script,str((ROOT/args.config).resolve())],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
