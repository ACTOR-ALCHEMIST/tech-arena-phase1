"""Conditional history-only residual; candidate positions and bases carry no role."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_two_stage import RelationBlock
from arena_history_data import TOKEN_WIDTH

CONFIGS={
 'history_mean':dict(width=64,attention=False,slate=False,history_dropout=.1,rank_weight=0.,seed=83871,scale=.35),
 'history_din':dict(width=64,attention=True,slate=False,history_dropout=.1,rank_weight=0.,seed=83871,scale=.35),
 'history_din_slate':dict(width=96,attention=True,slate=True,history_dropout=.15,rank_weight=0.,seed=83872,scale=.35),
 'history_din_rank':dict(width=96,attention=True,slate=True,history_dropout=.25,rank_weight=.1,seed=83873,scale=.35),
}
for _config in CONFIGS.values():_config['token_scaling']='fixed_v1'


class HistoryReranker(nn.Module):
    def __init__(self,mean,scale,config):
        super().__init__();self.config=config;w=config['width']
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        if config.get('token_scaling')=='fixed_v1':
            center=np.zeros(TOKEN_WIDTH,dtype='f');mult=np.ones(TOKEN_WIDTH,dtype='f')
            mult[:12]=3.;center[12:21]=1.;mult[12:21]=3.;center[21]=.5;mult[21]=2.
            self.register_buffer('token_center',torch.tensor(center));self.register_buffer('token_mult',torch.tensor(mult))
        self.query=nn.Sequential(nn.Linear(len(mean)+8,64),nn.SiLU(),nn.Linear(64,16))
        self.value=nn.Sequential(nn.Linear(TOKEN_WIDTH,w),nn.SiLU(),nn.Dropout(.1),nn.Linear(w,w),nn.SiLU())
        self.attention=nn.Sequential(nn.Linear(w+16,w),nn.SiLU(),nn.Linear(w,4)) if config['attention'] else None
        self.history_output=nn.Sequential(nn.Linear(w*4,w),nn.LayerNorm(w),nn.SiLU())
        self.blocks=nn.ModuleList([RelationBlock(w),RelationBlock(w)]) if config['slate'] else nn.ModuleList()
        self.context=nn.Sequential(nn.Linear(w*3,64),nn.SiLU())
        self.score=nn.Linear(w+64,1);nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)

    def corrections(self,x,p,tokens,visible,pair):
        p=p.clamp(1e-7,1-1e-7);click=p.sum(1).clamp(1e-6,1-1e-6);q=p/click[:,None];lp=q.log()
        top=lp.topk(2,dim=1).values;gap=top[:,0]-top[:,1]
        entropy=-(q*lp).sum(1)/np.log(10)
        rank=(p[:,:,None]>p[:,None,:]).float().mean(2)
        expand=lambda a:a[:,None].expand(-1,10)
        stats=torch.stack([p.log()/4,(lp-lp.mean(1,keepdim=True))/4,(lp-top[:,0,None])/4,q,rank,
                           expand(click),expand(entropy),expand(gap.clamp(max=8)/4)],2)
        query=self.query(torch.cat([((x-self.mean)/self.scale).clamp(-6,6),stats],2))
        mask=visible
        if self.training and self.config['history_dropout']:
            mask=mask&(torch.rand(mask.shape,device=mask.device)>=self.config['history_dropout'])
        if self.config.get('token_scaling')=='fixed_v1':tokens=(tokens-self.token_center)*self.token_mult
        value=self.value(tokens)
        if self.attention is None:
            weights=mask[:,None,:,None].float().expand(-1,10,-1,4)
        else:
            ctx=query[:,:,None].expand(-1,-1,20,-1)
            logits=self.attention(torch.cat([value,ctx],3)).masked_fill(~mask[:,None,:,None],-1e4)
            weights=logits.softmax(2)*mask[:,None,:,None]
        weights=weights/weights.sum(2,keepdim=True).clamp_min(1e-8)
        pooled=torch.einsum('bith,bitd->bihd',weights,value).flatten(2)
        z=self.history_output(pooled)
        for block in self.blocks:z=block(z,pair)
        aggregate=torch.cat([z.mean(1),z.max(1).values,z.std(1,unbiased=False)],1)
        ctx=self.context(aggregate)[:,None].expand(-1,10,-1)
        delta=self.score(torch.cat([z,ctx],2)).squeeze(2).tanh()
        delta=delta-delta.mean(1,keepdim=True)
        sparse=(1-visible.float().mean(1)).square()
        gate=.15+.85*torch.maximum(torch.exp(-gap/.75),.5*sparse)
        return self.config['scale']*delta*gate[:,None]*mask.any(1)[:,None]

    def forward(self,*inputs):
        p=inputs[1].clamp(1e-7,1-1e-7);click=p.sum(1,keepdim=True).clamp(1e-6,1-1e-6)
        choice=F.log_softmax(p.log()+self.corrections(*inputs),dim=1)
        return torch.cat([choice+click.log()-torch.log1p(-click),torch.zeros_like(click)],1)

    def probabilities(self,*inputs):return self(*inputs).softmax(1)[:,:10]
