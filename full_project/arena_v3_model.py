"""History and candidate Transformers with a larger numerical scoring tower."""
import torch
from torch import nn


def tower(widths,dropout=.1):
    layers=[]
    for a,b in zip(widths,widths[1:]):layers += [nn.Linear(a,b),nn.SiLU(),nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class HistoryTransformer(nn.Module):
    def __init__(self,config):
        super().__init__();d=config['width'];p=config.get('dropout',.1);extra=config.get('foundation_dims',0)
        from arena_neural_v2 import LargeSlateModel
        import numpy as np
        self.base=LargeSlateModel(112,np.zeros(112,np.float32),np.ones(112,np.float32),config['base_config'])
        self.train_base=config.get('unfreeze_base',False)
        self.base.requires_grad_(self.train_base)
        self.history_projection=nn.Linear(36,d)
        self.history_position=nn.Parameter(torch.randn(20,d)*.01)
        self.history=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,config['heads'],d*4,p,
            activation='gelu',batch_first=True,norm_first=True),config['history_layers'],enable_nested_tensor=False)
        self.query=nn.Linear(144+extra,d)
        self.cross=nn.MultiheadAttention(d,config['heads'],dropout=p,batch_first=True)
        self.slate=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,config['heads'],d*4,p,
            activation='gelu',batch_first=True,norm_first=True),config['slate_layers'],enable_nested_tensor=False)
        self.context=tower([374,128,config['context_width']],p)
        self.deep=tower([144+extra+config['context_width']]+config['hidden'],p)
        self.score=nn.Linear(config['hidden'][-1]+2*d,1)
        self.outside=nn.Sequential(tower([374,128,64],p),nn.Linear(64,1))
        self.register_buffer('foundation_mean',torch.zeros(extra));self.register_buffer('foundation_scale',torch.ones(extra))
        nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.outside[-1].weight);nn.init.zeros_(self.outside[-1].bias)

    def forward(self,event,history,candidate,foundation=None):
        mask=history[:,:,-1]<.5;empty=mask.all(1)
        # A finite dummy key handles empty histories; its readout is zeroed.
        mask=mask.clone();mask[:,0]=mask[:,0]&~empty
        h=self.history(self.history_projection(history)+self.history_position,src_key_padding_mask=mask)
        c=candidate
        if foundation is not None:
            f=((foundation-self.foundation_mean)/self.foundation_scale).clamp(-6,6)
            c=torch.cat([c,f],-1)
        query=self.query(c)
        read,_=self.cross(query,h,h,key_padding_mask=mask,need_weights=False)
        read=read.masked_fill(empty[:,None,None],0)
        slate=self.slate(query)
        context=self.context(event)[:,None].expand(-1,10,-1)
        score=self.score(torch.cat([self.deep(torch.cat([c,context],-1)),read,slate],-1)).squeeze(-1)
        self.base.eval()
        if self.train_base:base=self.base(candidate[:,:,:112])
        else:
            with torch.no_grad():base=self.base(candidate[:,:,:112])
        return base+.2*torch.cat([score,self.outside(event)],1)

    def probabilities(self,*args):return self(*args).softmax(-1)[:,:10]
