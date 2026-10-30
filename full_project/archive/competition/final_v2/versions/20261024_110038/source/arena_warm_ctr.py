"""Supervised real-exposure CTR fine tuning of auxiliary history embeddings."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ctr_data import load_cache,temporal_split,fit_priors
from run import timestamp,get_batch,predict_arrays
from arena_core import evaluate
from arena_sgns_features import embedding_model
from arena_report import LAB,note,write_json


class WarmCTR(nn.Module):
    def __init__(self,pretrained,priors,anchor=0.1,dropout=0.1,fine_tune=True):
        super().__init__();self.anchor=anchor;self.dropout=dropout
        uf=torch.tensor(pretrained['users']);vf=torch.tensor(pretrained['items']);dim=uf.shape[1]
        self.users=nn.Embedding.from_pretrained(uf.clone(),freeze=not fine_tune,padding_idx=0)
        self.items=nn.Embedding.from_pretrained(vf.clone(),freeze=not fine_tune,padding_idx=0)
        self.register_buffer('u0',uf);self.register_buffer('v0',vf)
        ctr,ub,ib,ku,ki=priors
        self.register_buffer('known_users',torch.tensor(ku));self.register_buffer('known_items',torch.tensor(ki))
        self.user_bias=nn.Embedding(len(ub),1,padding_idx=0);self.item_bias=nn.Embedding(len(ib),1,padding_idx=0)
        with torch.no_grad():
            self.user_bias.weight[:,0].copy_(torch.tensor(ub)*0.3);self.item_bias.weight[:,0].copy_(torch.tensor(ib)*0.3)
        self.register_buffer('ub0',self.user_bias.weight.detach().clone());self.register_buffer('ib0',self.item_bias.weight.detach().clone())
        self.global_bias=nn.Parameter(torch.tensor(float(np.log(ctr/(1-ctr)))))
        self.network=nn.Sequential(nn.Linear(18,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.linear=nn.Linear(18,1,bias=False);nn.init.zeros_(self.linear.weight)
        with torch.no_grad():self.linear.weight[0,1]=2;self.linear.weight[0,4]=2

    def forward(self,batch):
        uid=batch['users'];uid=torch.where(self.known_users[uid],uid,0)
        if self.training:uid=torch.where(torch.rand_like(uid,dtype=torch.float32)<self.dropout,0,uid)
        ids=torch.where(self.known_items[batch['items']],batch['items'],0)
        hist=torch.where(self.known_items[batch['history']],batch['history'],0)
        u=self.users(uid);v=self.items(ids);h=self.items(hist)
        un=u.norm(dim=1).clamp_min(1e-6);vn=v.norm(dim=2).clamp_min(1e-6);hn=h.norm(dim=2).clamp_min(1e-6)
        dot=(u[:,None,:]*v).sum(2);cos=dot/(un[:,None]*vn)
        hc=torch.bmm(v,h.transpose(1,2))/(vn[:,:,None]*hn[:,None,:]);mask=hist!=0;hc=hc*mask[:,None,:]
        mean=hc.sum(2)/mask.sum(1).clamp_min(1)[:,None];maximum=hc.max(2).values
        weight=torch.exp(-batch['age_hours']/72)*mask
        recent=(hc*weight[:,None,:]).sum(2)/weight.sum(1).clamp_min(1e-6)[:,None]
        match=(batch['items'][:,:,None]==batch['history'][:,None,:])&(batch['history'][:,None,:]!=0)
        repeats=match.sum(2).float()
        ctx=batch['context'][:,None,:].expand(-1,10,-1)
        ub=self.user_bias(uid).expand(-1,10);ib=self.item_bias(ids).squeeze(2)
        values=torch.stack([dot/8,cos,un[:,None].expand_as(dot)/5,vn/5,mean,maximum,recent,
                            torch.log1p(repeats),(repeats>0).float(),dot-dot.mean(1,keepdim=True),ub,ib],2)
        values=torch.cat([values,ctx],2)
        return self.global_bias+ub+ib+self.linear(values).squeeze(2)+self.network(values).squeeze(2)

    def penalty(self,batch):
        uid=batch['users'];iid=batch['items']
        return self.anchor*(F.mse_loss(self.users(uid),self.u0[uid])+F.mse_loss(self.items(iid),self.v0[iid])+
                            F.mse_loss(self.user_bias(uid),self.ub0[uid])+0.2*F.mse_loss(self.item_bias(iid),self.ib0[iid]))


def run(config):
    start=time.monotonic();torch.manual_seed(config.get('seed',42));torch.set_num_threads(2)
    device=torch.device('mps');cutoff=config.get('cutoff','2025-01-26')
    out=Path(config['output_dir']) if config.get('output_dir') else LAB/'experiments'/config['name'];out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    pre=embedding_model(cutoff,8,epochs=159,recipe={'epochs':159,'lr':0.01,'lr_decay':False,'reuse_experiment':'sgns8_const_e160_s42'})
    meta,data=load_cache('data/cache_v1','train')
    if config.get('evaluate',True):fit,valid=temporal_split(data,timestamp(cutoff))
    else:fit,valid=np.arange(len(data['users'])),np.empty(0,dtype=np.int64)
    priors=fit_priors(data,fit,meta['n_users'],meta['n_items'])
    model=WarmCTR(pre,priors,config.get('anchor',0.1),config.get('user_dropout',0.1),config.get('fine_tune',True)).to(device)
    embedding_parameters=list(model.users.parameters())+list(model.items.parameters())
    embedding_ids={id(p) for p in embedding_parameters}
    head=[p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer=torch.optim.AdamW([{'params':embedding_parameters,'lr':config.get('embedding_lr',0.0003),'weight_decay':0},
                                {'params':head,'lr':0.003,'weight_decay':0.001}])
    rng=np.random.default_rng(config.get('seed',42));records=[];best=-1;stale=0
    for epoch in range(1,config.get('epochs',8)+1):
        tick=time.monotonic();model.train();order=rng.permutation(fit);loss_sum=0
        for offset in range(0,len(order),2048):
            indices=order[offset:offset+2048];batch=get_batch(data,indices,device,labels=True)
            loss=F.binary_cross_entropy_with_logits(model(batch),batch['labels'])+model.penalty(batch)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            loss_sum+=loss.item()*len(indices)
        if not config.get('evaluate',True):
            row={'epoch':epoch,'train_loss':loss_sum/len(fit),'seconds':time.monotonic()-tick};records.append(row);print(json.dumps(row),flush=True)
            torch.save({'state_dict':model.state_dict(),'config':config,'cache_id':meta['cache_id']},out/'model.pt')
            (out/'training_result.json').write_text(json.dumps({'config':config,'epochs':records},indent=2))
            continue
        prediction=predict_arrays(model,data,valid,1024,device)
        row={'epoch':epoch,'train_loss':loss_sum/len(fit),**evaluate(data['labels'][valid],prediction),'seconds':time.monotonic()-tick}
        records.append(row);print(json.dumps(row),flush=True)
        save=(epoch==config.get('epochs',8)) if config.get('fixed_epochs') else row['auc']>best
        if save:
            best=row['auc'];stale=0
            torch.save({'state_dict':model.state_dict(),'config':config,'cache_id':meta['cache_id']},out/'model.pt')
            np.savez_compressed(out/'validation.npz',event_indices=valid,labels=data['labels'][valid],predictions=prediction)
        else:stale+=1
        (out/'training.json').write_text(json.dumps(records,indent=2))
        if not config.get('fixed_epochs') and stale>=3:break
    if not config.get('evaluate',True):return
    winner=records[-1] if config.get('fixed_epochs') else max(records,key=lambda row:row['auc'])
    result={'name':config['name'],'cutoff':cutoff,'method':'warm_ctr_mps',**{k:v for k,v in winner.items() if k not in ['epoch','seconds']},
            'best_epoch':winner['epoch'],'config':config,'epochs':records,'seconds':time.monotonic()-start}
    write_json(out/'result.json',result);note(f"真实曝光微调 {config['name']}：AUC {best:.6f}，anchor={config.get('anchor',0.1)}，最佳epoch {winner['epoch']}。",'experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()))
