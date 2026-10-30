"""The single shared feature recipe used by validation and final inference."""
import numpy as np
from arena_core import features,snapshot


def complete_features(data,indices,cutoff,config):
    x,names,_,_=features(data,indices,snapshot(cutoff))
    if config.get("latent_factors"):
        from arena_latent import latent_features
        for size in config["latent_factors"]:
            extra,cols=latent_features(data,indices,cutoff,size);x=np.concatenate([x,extra],1);names+=cols
    if config.get("context_features"):
        from arena_feedback import context_features
        extra,cols=context_features(data,indices,cutoff);x=np.concatenate([x,extra],1);names+=cols
    if config.get("sgns_factors"):
        from arena_sgns_features import embedding_features
        for size in config["sgns_factors"]:
            extra,cols=embedding_features(data,indices,cutoff,size,
                epochs=config.get("sgns_epochs",{}).get(str(size),80 if size<=8 else 56),
                recipe=config.get("sgns_recipes",{}).get(str(size)))
            x=np.concatenate([x,extra],1);names+=cols
    if config.get("foldin"):
        from arena_foldin import foldin_features
        extra,cols=foldin_features(data,indices,cutoff,config);x=np.concatenate([x,extra],1);names+=cols
    if config.get('alignment'):
        from arena_alignment import aligned_features
        extra,cols=aligned_features(data,indices,cutoff,config);x=np.concatenate([x,extra],1);names+=cols
    return x,names
