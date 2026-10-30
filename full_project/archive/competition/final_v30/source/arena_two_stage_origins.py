"""Second reranking campaign: first-query views and separate choice calibration."""
from arena_ood import ROOT
import arena_two_stage_campaign as campaign

if __name__=='__main__':
    campaign.OUT=ROOT/'competition/two_stage_origins'
    campaign.FINAL=ROOT/'competition/final_v22'
    campaign.NAMES=['origin_edge_mlp_choice','origin_edge_transformer_choice','origin_choice_only']
    campaign.run()
