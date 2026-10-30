"""Download separate, revision-pinned candidate foundations; preserve V2/LLM sources."""
import hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
out=Path('competition/v3/model_sources');out.mkdir(parents=True,exist_ok=True)
for repo in ['Qwen/Qwen3-0.6B','Qwen/Qwen2.5-0.5B','Qwen/Qwen2.5-1.5B']:
    target=out/(repo.split('/')[-1]+'.json')
    if target.exists():print('Retained',target,flush=True);continue
    revision=HfApi().model_info(repo).sha
    folder=Path(snapshot_download(repo,revision=revision,cache_dir='.tools/hf-generative',
        allow_patterns=['*.safetensors','*.safetensors.index.json','config.json'],max_workers=2)).resolve()
    info=dict(repo_id=repo,revision=revision,local_path=str(folder),
        weight_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in folder.glob('*.safetensors')})
    target.write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
