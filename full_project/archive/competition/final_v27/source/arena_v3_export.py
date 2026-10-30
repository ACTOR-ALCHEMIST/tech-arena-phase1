"""Refit the selected recipe and verify the best standalone submission artifact."""
import csv,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from arena_v3 import ROOT,OUT,RECIPES,prepare,train,arrays,predict,guard,sha
from arena_v3_iterate import record,EXTRAS
from arena_report import write_json

FINAL=ROOT/'competition/final_v3'


def export():
    from arena_finalize import blend,refit_tabular,infer_model,export_csv
    from ctr_data import load_cache
    before=guard();selection=json.loads((OUT/'selection.json').read_text());spec=selection['selected']
    FINAL.mkdir(parents=True,exist_ok=True)
    if (FINAL/'manifest.json').exists():return
    active=[(name,w) for name,w in zip(selection['sources'][1:],spec['weights'][1:]) if w>0]
    predictions=[];weights=[];models=[]
    _,evaluation=load_cache('data/cache_v1','evaluation')
    if spec['weights'][0]>0:
        # Preserve the exact previously accepted deployed probabilities.
        p=[];ids=evaluation['rowids'].ravel()
        with (ROOT/'competition/final/submission.csv').open(newline='') as f:
            reader=csv.reader(f);next(reader)
            for i,(rid,value) in enumerate(reader):assert rid==ids[i].decode();p.append(float(value))
        assert len(p)==len(ids)
        predictions.append(np.array(p).reshape(-1,10));weights.append(spec['weights'][0])
        shutil.copy2(ROOT/'competition/final/submission.csv',FINAL/'frozen_v2.csv')
    for name,w in active:
        primary,_=record(name,'2025-01-26')
        if name in RECIPES and primary['best_epoch']>0:
            folder=prepare('2025-01-27');foundation=RECIPES[name]['foundation']
            if foundation:
                subprocess.run([sys.executable,'-u','arena_v3_foundation.py','--name',foundation,'--cutoff','2025-01-27','--layers',str(RECIPES[name].get('foundation_layers',12))],cwd=ROOT,check=True)
            fitted=train(name,'2025-01-27',primary['best_epoch'])
            path=OUT/'runs'/name/'2025-01-27';p=np.load(path/'validation.npz')['predictions']
            inputs=arrays(folder,'valid',foundation)
            input_hashes={key:sha(folder/('valid_'+(foundation if key=='foundation' else key)+'.npy')) for key in inputs}
            models.append(dict(name=name,type='history_transformer',path=str(path.relative_to(ROOT)),weight=w,model_sha256=sha(path/'model.pt'),
                cached_prediction_sha256=sha(path/'validation.npz'),input_sha256=input_hashes,
                foundation_metadata=json.loads((folder/(foundation+'_metadata.json')).read_text()) if foundation else None))
        else:
            path=FINAL/'models'/name;path.mkdir(parents=True,exist_ok=True)
            reused_initial=name in RECIPES
            if not (path/'model.pt').exists():
                if reused_initial:
                    # The selected epoch-0 residual is identically zero. Its
                    # exact effective scorer is the inherited numerical base.
                    import torch
                    from arena_v3 import base_checkpoint
                    original=torch.load(OUT/'runs'/name/'2025-01-26/model.pt',map_location='cpu',weights_only=True)['state_dict']
                    for key in ['score.weight','score.bias','outside.1.weight','outside.1.bias']:
                        assert torch.count_nonzero(original[key])==0
                    source_model=base_checkpoint('2025-01-27').parent
                    for file in ['model.pt','recipe.json']:shutil.copy2(source_model/file,path/file)
                else:refit_tabular(primary,path)
            p=infer_model(path,evaluation)
            np.save(path/'evaluation_predictions.npy',p)
            models.append(dict(name=name,type='tabular',path=str(path.relative_to(ROOT)),weight=w,model_sha256=sha(path/'model.pt'),
                cached_prediction_sha256=sha(path/'evaluation_predictions.npy'),initial_checkpoint_reused=reused_initial))
        predictions.append(p);weights.append(w)
    if not active:
        shutil.copy2(ROOT/'competition/final/submission.csv',FINAL/'submission.csv')
        prediction=predictions[0]
    else:
        prediction=blend(predictions,weights,spec['kind']);export_csv(FINAL/'submission.csv',evaluation,prediction)
    np.save(FINAL/'predictions.npy',prediction)
    source=FINAL/'source';source.mkdir(exist_ok=True)
    files=['arena_v3.py','arena_v3_model.py','arena_v3_foundation.py','arena_v3_iterate.py','arena_v3_export.py','arena_v3_download.py','test_arena_v3.py','requirements-v3.txt',
        'arena_neural_v2.py','arena_tabular.py','arena_finalize.py','arena_core.py','arena_models.py','arena_features.py','arena_sgns_features.py',
        'arena_latent.py','arena_feedback.py','arena_alignment.py','arena_validation.py','generative_data.py','ctr_data.py','run.py','arena_report.py']
    for file in files:shutil.copy2(ROOT/file,source/file)
    assert guard()==before
    manifest=dict(method='v3_two_cutoff_selected_ensemble',selection=spec,sources=selection['sources'],models=models,
        baseline_weight=spec['weights'][0],baseline_sha256=before['submission.csv'],incumbent_digest=before,
        csv='competition/final_v3/submission.csv',sha256=sha(FINAL/'submission.csv'),rows=529390,prediction_sha256=sha(FINAL/'predictions.npy'),
        source_sha256={f:sha(ROOT/f) for f in files},selection_sha256=sha(OUT/'selection.json'),
        environment=json.loads((OUT/'environment.json').read_text()),
        validation_26=dict(auc=spec['auc_26']),validation_25=dict(auc=spec['auc_25']),
        official_evaluation_labels_available=False,at_least_three_new_iterations_completed=len(selection['candidates'])>=3)
    write_json(FINAL/'manifest.json',manifest)


def verify():
    import torch
    from arena_v3_model import HistoryTransformer
    from ctr_data import load_cache
    from run import check_submission
    from arena_finalize import blend
    manifest=json.loads((FINAL/'manifest.json').read_text());assert guard()==manifest['incumbent_digest']
    assert sha(FINAL/'submission.csv')==manifest['sha256'];assert sha(OUT/'selection.json')==manifest['selection_sha256']
    assert sha(FINAL/'predictions.npy')==manifest['prediction_sha256']
    for f,digest in manifest['source_sha256'].items():assert sha(ROOT/f)==digest
    check_submission(FINAL/'submission.csv','evaluation.zip')
    _,raw=load_cache('data/cache_v1','evaluation');ids=raw['rowids'].ravel();expected=np.load(FINAL/'predictions.npy').ravel();count=0;rounding=0.
    with (FINAL/'submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['rowid','predicted_probability']
        for i,(rid,p) in enumerate(reader):
            assert rid==ids[i].decode();rounding=max(rounding,abs(float(p)-float(expected[i])));count+=1
    assert count==529390 and rounding<1e-9
    errors=[];parts=[];weights=[]
    if manifest['baseline_weight']:
        assert sha(FINAL/'frozen_v2.csv')==manifest['baseline_sha256']
        with (FINAL/'frozen_v2.csv').open(newline='') as f:
            reader=csv.reader(f);next(reader);parts.append(np.array([float(r[1]) for r in reader]).reshape(-1,10))
        weights.append(manifest['baseline_weight'])
    for row in manifest['models']:
        path=ROOT/row['path'];assert sha(path/'model.pt')==row['model_sha256']
        if row['type']=='history_transformer':
            assert sha(path/'validation.npz')==row['cached_prediction_sha256']
            saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True);model=HistoryTransformer(saved['config']).to('mps');model.load_state_dict(saved['state_dict']);model.eval()
            values=arrays(OUT/'data'/'2025-01-27','valid',saved['config']['foundation']);n=len(values['event']);original=np.load(path/'validation.npz')['predictions']
            for key,digest in row['input_sha256'].items():
                file='valid_'+(saved['config']['foundation'] if key=='foundation' else key)+'.npy'
                assert sha(OUT/'data'/'2025-01-27'/file)==digest
            for start in [0,512*17,((n-1)//512)*512]:
                block={k:v[start:start+512] for k,v in values.items()};p=predict(model,block)
                error=float(np.max(np.abs(p-original[start:start+len(p)])));assert error<2e-5;errors.append(error)
        else:
            assert sha(path/'evaluation_predictions.npy')==row['cached_prediction_sha256']
            from arena_tabular import build_model,predict as numeric_predict
            from arena_finalize import feature_cache
            recipe=json.loads((path/'recipe.json').read_text());x=np.load(feature_cache(recipe['config'],'evaluation'),mmap_mode='r')
            saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True);s=saved['state_dict'];m=build_model(saved['width'],s['mean'].numpy(),s['scale'].numpy(),saved['config']);m.load_state_dict(s);m.to('mps')
            original=np.load(path/'evaluation_predictions.npy')
            # Original tabular inference uses complete 2048-event batches.
            for start in [0,2048*7,((len(original)-1)//2048)*2048]:
                p=numeric_predict(m,x[start*10:(start+2048)*10],torch.device('mps'))
                error=float(np.max(np.abs(p-original[start:start+len(p)])));assert error<2e-5;errors.append(error)
        parts.append(original);weights.append(row['weight'])
    composed=blend(parts,weights,'logit')
    composition_error=float(np.max(np.abs(composed.ravel()-expected)));assert composition_error<1e-12
    assert guard()==manifest['incumbent_digest']
    result=dict(passed=True,rows=count,csv_sha256=sha(FINAL/'submission.csv'),manifest_sha256=sha(FINAL/'manifest.json'),
        csv_rounding_max_error=rounding,fresh_process_new_model_max_error=max(errors,default=0.),composition_max_error=composition_error,incumbent_unchanged=True,
        training_iterations_completed=len(json.loads((OUT/'selection.json').read_text())['candidates']))
    write_json(FINAL/'acceptance.json',result)
    write_report(manifest,result)
    print(json.dumps(result,indent=2),flush=True)


def write_report(manifest,acceptance):
    selection=json.loads((OUT/'selection.json').read_text());lines=['# V3：Transformer 与不同基模的五轮迭代','',
        '旧 V2 和上一轮 Qwen3.5 LoRA 模型、CSV 均保留。本轮在同一完整时间验证集比较，Jan25 使用 Jan26 选定的轮数及固定学习率计划。',
        '', '| 迭代 | Jan26 AUC | Jan25 AUC | 可训练参数 | 选中轮数 |','|---|---:|---:|---:|---:|']
    for row in selection['candidates']:
        a=row['primary'];b=row['secondary'];lines.append(f"| {row['name']} | {a['auc']:.6f} | {b['auc']:.6f} | {a.get('trainable_parameters',a.get('parameters',0)):,} | {a['best_epoch']} |")
    lines += ['', '选中轮数为 0 表示新增结构训练后没有超过继承的初始评分器，验证最佳保留初始检查点；它不代表大模型取得提升。']
    lines += ['', '| 冻结底座 | 完整模型参数 | 使用层数 | 使用的 Transformer 层参数 |', '|---|---:|---:|---:|']
    for name in ['Qwen3-0.6B','Qwen2.5-1.5B']:
        m=json.loads((OUT/'data'/'2025-01-26'/(name+'_metadata.json')).read_text())['source']
        lines.append(f"| {name} | {m['full_parameters']:,} | {m['used_layers']}/{m['full_layers']} | {m['used_transformer_parameters']:,} |")
    old=selection['incumbent'];new=selection['selected']
    lines += ['',f"最终选择：Jan26 {new['auc_26']:.9f}，Jan25 {new['auc_25']:.9f}；原 V2 分别 {old['auc_26']:.9f}、{old['auc_25']:.9f}。",'',
        '模型容量较大不保证 AUC 上升；只有在两个切点和平均群体加权指标均改进的混合配方才能替换最终交付。若均未改进，最终 CSV 保留原最优预测。','',
        '预训练分支实际加载官方 Qwen3-0.6B 与 Qwen2.5-1.5B 权重，分别使用前 12 层／前 6 层的冻结表示，经固定随机软输入投影、32 维输出投影送入历史 Transformer 评分头。本轮不微调语言底座，也不把随机输入投影称作预训练语义编码器。训练评分头使用 18 万事件；入选的新模型正式 refit 使用最近窗口全部 293,993 个事件。','',
        '这些尝试对已有特征、随机种子和模型池有选择依赖，本地改进不能保证排行榜提升。正式评测集无标签。','',
        '[可直接提交的最优 CSV](../final_v3/submission.csv)；[验收](../final_v3/acceptance.json)；[配方比较](selection.json)。','',
        '官方来源：[Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B)、[Transformers 向量输入](https://huggingface.co/docs/transformers/model_doc/qwen3)。','',
        '复现／继续：`python3 arena_v3_iterate.py`。已有完成结果自动保留；新实验请使用新名字。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from arena_report import note
    note(f"V3 五轮结构／基模迭代与双切点复核完成，交付 CSV {acceptance['rows']:,} 行，Jan26 AUC {new['auc_26']:.6f}，Jan25 AUC {new['auc_25']:.6f}，原最优保留。",'v3')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['export','verify']);a=p.parse_args()
    {'export':export,'verify':verify}[a.command]()
