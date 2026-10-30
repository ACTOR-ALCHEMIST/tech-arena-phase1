"""Use all actual clicked slates, beyond the recent-two-day MAP update."""
import json
import time
from arena_ood import ROOT,write_json
import arena_conditional_sphere as experiment

experiment.OUT=ROOT/'competition/conditional_full'
experiment.RECIPE=dict(experiment.RECIPE,version=2,fit_window='full_prefix',prior_scales=[1.,10.,50.])


if __name__=='__main__':
    experiment.OUT.mkdir(parents=True,exist_ok=True)
    for day in ['26','25']:
        write_json(experiment.OUT/'state.json',dict(stage='training',day=day,updated=time.time()))
        experiment.run('2025-01-'+day)
    experiment.select()
