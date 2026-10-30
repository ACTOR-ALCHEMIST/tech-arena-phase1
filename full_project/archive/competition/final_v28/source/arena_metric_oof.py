"""Fit both first-stage scores and geometry from their own earlier OOF frames."""
from arena_ood import ROOT
import arena_metric_rerank as experiment

if __name__=='__main__':
    experiment.OUT=ROOT/'competition/metric_oof'
    experiment.GEOMETRY_MODE='aligned_oof'
    experiment.run()
