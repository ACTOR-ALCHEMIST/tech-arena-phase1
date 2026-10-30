"""Small MPS numerical heads on strictly out-of-time representation features."""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_core import LAB,ROOT,evaluate
from arena_report import note,write_json


class SlateMLP(nn.Module):
    def __init__(self,width,mean,scale,loss="event_ce",dropout=0.1,architecture="point"):
        super().__init__();self.loss=loss
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.context=nn.Sequential(nn.Linear(width*3,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU()) if architecture=='context' else None
        self.item=nn.Sequential(nn.Linear(width+(32 if self.context is not None else 0),96),nn.SiLU(),nn.Dropout(dropout),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1))
        self.outside=nn.Sequential(nn.Linear(width*3,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,x):
        x=((x-self.mean)/self.scale).clamp(-6,6)
        aggregate=torch.cat([x.mean(1),x.max(1).values,x.std(1,unbiased=False)],1)
        values=torch.cat([x,self.context(aggregate)[:,None,:].expand(-1,x.shape[1],-1)],2) if self.context is not None else x
        scores=self.item(values).squeeze(2)
        outside=self.outside(aggregate)
        return torch.cat([scores,outside],1) if self.loss=='event_ce' else scores

    def probabilities(self,x):
        z=self(x)
        return torch.softmax(z,1)[:,:10] if self.loss=='event_ce' else torch.sigmoid(z)


def normalizer(x,constant_policy='legacy_clip'):
    total=np.zeros(x.shape[1],np.float64);squared=total.copy()
    for offset in range(0,len(x),100000):
        b=np.asarray(x[offset:offset+100000]);total+=b.sum(0,dtype=np.float64);squared+=(b*b).sum(0,dtype=np.float64)
    mean=total/len(x);variance=squared/len(x)-mean*mean
    scale=np.sqrt(np.maximum(variance,1e-6))
    if constant_policy=='zero':scale=np.where(variance<1e-8,1e12,scale)
    return mean.astype(np.float32),scale.astype(np.float32)


@torch.inference_mode()
def predict(model,x,device,batch_events=2048):
    model.eval();parts=[]
    for offset in range(0,len(x)//10,batch_events):
        data=np.array(x[offset*10:(offset+batch_events)*10]).reshape(-1,10,x.shape[1])
        parts.append(model.probabilities(torch.tensor(data,device=device)).cpu().numpy())
    return np.concatenate(parts)


def run(config):
    torch.manual_seed(config.get('seed',42));torch.set_num_threads(2);device=torch.device('mps');started=time.monotonic()
    base=config.get('feature_reference') or json.loads((LAB/'experiments'/config['source_experiment']/'result.json').read_text())
    if config.get('aligned_coordinates'):
        from arena_models import feature_dataset
        fc=copy.deepcopy(base['config']);fc['alignment']={'factors':8}
        folder=feature_dataset(base['cutoff'],fc.get('train_start','2025-01-22'),fc.get('max_events',80000),fc.get('data_seed',42),
            fc.get('latent_factors'),fc.get('context_features',False),fc.get('sgns_factors'),fc.get('sgns_epochs'),
            fc.get('sgns_recipes'),fc.get('rolling',False),fc.get('foldin'),fc['alignment'])
        base=dict(base,config=fc,feature_dataset=str(folder.relative_to(ROOT)))
    folder=ROOT/base['feature_dataset'];labels=np.load(folder/'labels.npz')
    x=np.load(folder/'train.npy',mmap_mode='r');v=np.load(folder/'valid.npy',mmap_mode='r')
    y=labels['train'].reshape(-1,10);vy=labels['valid'].reshape(-1,10)
    config.setdefault('constant_policy','zero')
    mean,scale=normalizer(x,config['constant_policy']);loss_name=config.get('loss','event_ce')
    model=SlateMLP(x.shape[1],mean,scale,loss_name,config.get('dropout',0.1),config.get('architecture','point')).to(device)
    ema=copy.deepcopy(model).eval() if config.get('ema') else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',0.001),weight_decay=config.get('weight_decay',0.01))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.get('lr_schedule_epochs',config.get('epochs',30)),eta_min=0.0001)
    out=LAB/'experiments'/config['name'];out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2));rng=np.random.default_rng(config.get('seed',42))
    records=[];best=-1;stale=0
    for epoch in range(1,config.get('epochs',30)+1):
        tick=time.monotonic();model.train();order=rng.permutation(len(y));summed=0
        for offset in range(0,len(order),1024):
            chosen=order[offset:offset+1024];rows=(chosen[:,None]*10+np.arange(10)).ravel()
            bx=torch.tensor(np.array(x[rows]).reshape(-1,10,x.shape[1]),device=device)
            by=torch.tensor(y[chosen],dtype=torch.float32,device=device);z=model(bx)
            if loss_name=='event_ce':
                target=torch.where(by.sum(1)>0,by.argmax(1),10)
                loss=F.cross_entropy(z,target)
            else:loss=F.binary_cross_entropy_with_logits(z,by)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();summed+=loss.item()*len(chosen)
            if ema is not None:
                with torch.no_grad():
                    for target,parameter in zip(ema.parameters(),model.parameters()):target.mul_(config['ema']).add_(parameter,alpha=1-config['ema'])
        scheduler.step();evaluated=ema if ema is not None else model;prediction=predict(evaluated,v,device)
        row={'epoch':epoch,'train_loss':summed/len(y),**evaluate(vy,prediction),'seconds':time.monotonic()-tick}
        records.append(row);print(json.dumps(row),flush=True)
        save=(epoch==config.get('epochs',30)) if config.get('fixed_epochs') else row['auc']>best
        if save:
            best=row['auc'];stale=0
            torch.save({'state_dict':evaluated.state_dict(),'width':x.shape[1],'loss':loss_name,'config':config},out/'model.pt')
            np.savez_compressed(out/'validation.npz',event_indices=labels['validation_indices'],labels=vy,predictions=prediction)
        else:stale+=1
        (out/'training.json').write_text(json.dumps(records,indent=2))
        if not config.get('fixed_epochs') and stale>=5:break
    winner=records[-1] if config.get('fixed_epochs') else max(records,key=lambda row:row['auc'])
    result={'name':config['name'],'cutoff':base['cutoff'],'method':'tabular_mlp_mps',
            **{k:v for k,v in winner.items() if k not in ['epoch','seconds']},'best_epoch':winner['epoch'],
            'config':config,'epochs':records,'feature_dataset':base['feature_dataset'],
            'source_experiment':base['name'],'feature_config':base['config'],'seconds':time.monotonic()-started}
    write_json(out/'result.json',result);note(f"前向特征神经评分 {config['name']}：{loss_name}，AUC {winner['auc']:.6f}，最佳epoch {winner['epoch']}。",'experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()))
