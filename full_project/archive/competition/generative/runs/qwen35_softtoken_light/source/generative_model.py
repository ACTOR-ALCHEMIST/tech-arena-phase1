"""Qwen text backbone with learned soft inputs and eleven label verbalizers.

The output rows start from the pretrained token embeddings for A..K. We score
only those rows at the final position, avoiding a full-vocabulary tensor at
every input position. Class K is the no-click outcome. No KV cache is reused
across events or training steps.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx.utils import tree_flatten


def validate_label_ids(tokenizer,labels):
    encoded=[tokenizer.encode(t,add_special_tokens=False) for t in labels]
    if len(encoded)!=11 or any(len(ids)!=1 for ids in encoded):
        raise ValueError('Exactly eleven single-token labels are required')
    result=[ids[0] for ids in encoded]
    if len(set(result))!=11:raise ValueError('Label tokens must be distinct')
    return result


class GenerativeCTR(nn.Module):
    def __init__(self,backbone,tokenizer,dimensions,config,prior):
        super().__init__()
        self.backbone=backbone
        self.backbone.freeze()
        count=config.get('lora_layers',4)
        if not 0<=count<=len(backbone.layers):raise ValueError('Invalid LoRA layer count')
        if count:
            linear_to_lora_layers(self.backbone,count,dict(rank=config.get('lora_rank',8),
                scale=config.get('lora_scale',16.),dropout=0.))
        if config.get('gradient_checkpointing',True):
            from mlx_lm.tuner.trainer import grad_checkpoint
            layer=backbone.layers[0]
            if not getattr(type(layer),'_ctr_grad_checkpointed',False):
                grad_checkpoint(layer)
                setattr(type(layer),'_ctr_grad_checkpointed',True)
        self.label_ids=validate_label_ids(tokenizer,config['label_tokens'])
        original=backbone.model.embed_tokens(mx.array(self.label_ids))
        mx.eval(original)
        self.input_dtype=original.dtype
        self.width=original.shape[-1]
        self.token_scale=float(mx.sqrt(mx.mean(original.astype(mx.float32)**2)))
        # Frozen identifiers bind each candidate soft token to its output label.
        self.candidate_labels=original[:10].astype(mx.float32)
        query_ids=tokenizer.encode('Answer:',add_special_tokens=False)
        self.answer_vector=backbone.model.embed_tokens(mx.array([query_ids[-1]]))[0].astype(mx.float32)
        self.event_projection=nn.Linear(dimensions['event'],self.width)
        self.history_projection=nn.Linear(dimensions['history'],self.width)
        self.candidate_projection=nn.Linear(dimensions['candidate'],self.width)
        self.token_norm=nn.RMSNorm(self.width)
        self.type_vectors=mx.random.normal((3,self.width))*self.token_scale*.1
        self.output=nn.Linear(self.width,11,bias=False)
        rows=original.astype(mx.float32)
        self.output.weight=rows/mx.maximum(mx.linalg.norm(rows,axis=1,keepdims=True),1e-6)
        self.class_bias=mx.log(mx.array(prior,dtype=mx.float32))
        self.freeze(keys=['candidate_labels','answer_vector'],recurse=False)
        trainable=[name for name,_ in tree_flatten(self.trainable_parameters())]
        assert all(not n.startswith('backbone.') or n.endswith(('lora_a','lora_b')) for n in trainable),trainable

    def __call__(self,event,history,candidate):
        e=self.token_norm(self.event_projection(event))*self.token_scale+self.type_vectors[0]
        h=self.token_norm(self.history_projection(history))*self.token_scale+self.type_vectors[1]
        c=self.token_norm(self.candidate_projection(candidate))*self.token_scale+self.type_vectors[2]+self.candidate_labels
        answer=mx.broadcast_to(self.answer_vector[None,None,:],(event.shape[0],1,self.width))
        inputs=mx.concatenate([e[:,None,:],h,c,answer],axis=1).astype(self.input_dtype)
        dummy=mx.zeros(inputs.shape[:2],dtype=mx.int32)
        hidden=self.backbone.model(dummy,input_embeddings=inputs)[:,-1,:].astype(mx.float32)
        return .1*self.output(hidden)+self.class_bias

    def probabilities(self,event,history,candidate):
        return mx.softmax(self(event,history,candidate),axis=-1)[:,:10]
