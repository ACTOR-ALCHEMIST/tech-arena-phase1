"""Measure the real local model before choosing a larger training recipe."""
import argparse
import json
import time
from functools import partial
from pathlib import Path

from generative import load_model, predict
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map


def main():
    p=argparse.ArgumentParser();p.add_argument('--compile',action='store_true');a=p.parse_args()
    out=Path('competition/generative/runs/qwen35_softtoken_light')
    config=json.loads((out/'config.json').read_text());data=dict(np.load(out/'inputs.npz'))
    mx.set_cache_limit(256*2**20)
    model,_=load_model(config,data);model.load_weights(str(out/'adapters.safetensors'),strict=False)
    mx.eval(model.parameters())
    for batch in (4,8,16,32):
        tick=time.monotonic();value=predict(model,data,batch_size=batch,limit=64)
        print(json.dumps(dict(stage='inference',batch=batch,events=64,seconds=time.monotonic()-tick,
                              peak_memory_gib=mx.get_peak_memory()/2**30)),flush=True)
        del value;mx.clear_cache()
    if not a.compile:return
    model.train();optimizer=optim.AdamW(learning_rate=1e-4)
    optimizer.init(model.trainable_parameters())
    def objective(m,e,h,c,y):return nn.losses.cross_entropy(m(e,h,c),y,reduction='mean')
    value_grad=nn.value_and_grad(model,objective)
    state=[model.state,optimizer.state,mx.random.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(e,h,c,y):
        loss,grad=value_grad(model,e,h,c,y)
        norm=mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(grad)))
        grad=tree_map(lambda g:g*mx.minimum(1.,1./mx.maximum(norm,1e-8)),grad)
        optimizer.update(model,grad)
        return loss,norm
    for n in range(6):
        chosen=slice(n*4,(n+1)*4)
        inputs=[mx.array(data['train_'+k][chosen]) for k in ('event','history','candidate','target')]
        tick=time.monotonic();loss,norm=step(*inputs);mx.eval(loss,norm,state)
        print(json.dumps(dict(stage='compiled_train',step=n,loss=float(loss),seconds=time.monotonic()-tick,
                              peak_memory_gib=mx.get_peak_memory()/2**30)),flush=True)


if __name__=='__main__':main()
