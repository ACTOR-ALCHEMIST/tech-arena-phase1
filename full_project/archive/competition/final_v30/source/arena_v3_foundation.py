"""Frozen, real HF pretrained prefixes applied to context/candidate soft tokens.

The input projections are fixed random maps, not pretrained semantic encoders.
Only the downstream history Transformer is supervised. Used depth and parameter
counts are recorded explicitly; these are not full-model LoRA experiments.
"""
import argparse,json,time,math
from pathlib import Path
import numpy as np
from arena_v3 import OUT,arrays,prepare,sha,guard
from arena_report import write_json


def load_foundation(name,layers=12):
    import torch
    from transformers import AutoModel
    source=json.loads((OUT/'model_sources'/(name+'.json')).read_text())
    for file,digest in source['weight_sha256'].items():assert sha(Path(source['local_path'])/file)==digest
    model,loading=AutoModel.from_pretrained(source['local_path'],torch_dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
    if loading['missing_keys']:raise ValueError('Missing pretrained parameters: '+repr(loading['missing_keys']))
    full_layers=len(model.layers);full_parameters=sum(p.numel() for p in model.parameters())
    model.layers=model.layers[:layers];model.config.num_hidden_layers=layers
    source=dict(source,used_layers=layers,full_layers=full_layers,full_parameters=full_parameters,loading_info=loading,
        used_transformer_parameters=sum(p.numel() for layer in model.layers for p in layer.parameters()))
    model.requires_grad_(False);model.eval().to('mps')
    width=model.config.hidden_size
    token_scale=float(model.embed_tokens.weight[:1024].float().square().mean().sqrt())
    rng=torch.Generator().manual_seed(4126)
    projections=[(torch.randn(a,b,generator=rng)/math.sqrt(a)).to('mps') for a,b in [(74,width),(144,width),(width,32)]]
    return model,projections,token_scale,source


def features(model,projections,scale,values,start,end):
    import torch
    from torch.nn import functional as F
    with torch.inference_mode():
        e=torch.from_numpy(np.array(values['event'][start:end,:38])).to('mps')
        h=torch.from_numpy(np.array(values['history'][start:end])).to('mps')
        mask=h[:,:,-1:];pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        context=torch.cat([e,pooled],-1).repeat_interleave(10,dim=0)
        c=torch.from_numpy(np.array(values['candidate'][start:end])).to('mps').reshape(-1,144)
        a=context@projections[0];b=c@projections[1]
        a=F.layer_norm(a,(a.shape[-1],));b=F.layer_norm(b,(b.shape[-1],))
        tokens=torch.stack([a,b],dim=1).to(torch.float16)*scale
        out=model(inputs_embeds=tokens,use_cache=False).last_hidden_state[:,-1].float()
        out=out/out.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        out=(out@projections[2]).cpu().numpy().reshape(-1,10,32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite foundation features')
    return out


def extract(name,cutoff,layers=12,benchmark=False):
    folder=prepare(cutoff);target=folder/(name+'_metadata.json')
    if target.exists() and not benchmark:return
    before=guard();model,projection,scale,source=load_foundation(name,layers)
    if benchmark:
        values=arrays(folder,'train')
        for batch in [32,128,256]:
            features(model,projection,scale,values,0,batch)
            tick=time.monotonic();features(model,projection,scale,values,0,batch)
            print(json.dumps(dict(name=name,used_layers=layers,events=batch,seconds=time.monotonic()-tick)),flush=True)
        return
    started=time.monotonic();batch=128
    identity=dict(source=source,data_metadata_sha256=sha(folder/'metadata.json'),batch_events=batch,
        input_tokens_per_candidate=2,projection_seed=4126,dimensions=32,method='frozen_pretrained_prefix_fixed_soft_projection')
    for phase in ['train','valid']:
        values=arrays(folder,phase);n=len(values['event']);path=folder/(phase+'_'+name+'.npy');progress=folder/(phase+'_'+name+'_progress.json')
        done=0
        if path.exists() and progress.exists():
            old=json.loads(progress.read_text());assert old['identity']==identity;done=old['completed']
            output=np.lib.format.open_memmap(path,mode='r+')
        else:output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,10,32))
        for start in range(done,n,batch):
            end=min(start+batch,n);output[start:end]=features(model,projection,scale,values,start,end)
            if end%2048==0 or end==n:
                output.flush();write_json(progress,dict(identity=identity,completed=end,events=n))
                print(json.dumps(dict(stage='foundation_features',name=name,cutoff=cutoff,phase=phase,completed=end,events=n,seconds=time.monotonic()-started)),flush=True)
        del output,values
    assert guard()==before
    write_json(target,dict(**identity,seconds=time.monotonic()-started,
        feature_sha256={phase:sha(folder/(phase+'_'+name+'.npy')) for phase in ['train','valid']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['Qwen3-0.6B','Qwen2.5-0.5B','Qwen2.5-1.5B']);p.add_argument('--cutoff',default='2025-01-26')
    p.add_argument('--layers',type=int,default=12);p.add_argument('--benchmark',action='store_true');a=p.parse_args()
    extract(a.name,a.cutoff,a.layers,a.benchmark)
