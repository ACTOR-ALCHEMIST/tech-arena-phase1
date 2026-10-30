"""Leaf-wise histogram boosting controls; raw matrices freed after binning."""
import fcntl
import gc
import json
import shutil
import time
import numpy as np
import arena_core
import lightgbm as lgb
from arena_ood import ROOT, OUT, guard, sha, write_json
from arena_graph_trees import matrix, neural_logit


def params(config):
    return dict(objective='binary', metric='auc', num_leaves=config.get('leaves', 31),
                max_depth=8, learning_rate=.04, min_data_in_leaf=200, lambda_l2=30,
                feature_fraction=.9, bagging_fraction=.8, bagging_freq=1,
                max_bin=127, num_threads=2, seed=2026, deterministic=True,
                force_col_wise=True, verbosity=-1, boost_from_average=not config.get('neural_offset', False))


def fit_binned(config, training, validation=None, fixed_epochs=None):
    callbacks = [lgb.log_evaluation(100)]
    if fixed_epochs is None:
        callbacks.append(lgb.early_stopping(100, first_metric_only=True))
    return lgb.train(params(config), training, num_boost_round=config.get('iterations', 1500) if fixed_epochs is None else fixed_epochs,
                     valid_sets=[validation] if validation is not None else None, callbacks=callbacks)


def train(name, cutoff, fixed_epochs, overrides):
    from arena_ood import arrays
    from arena_validation import importance_weights, weighted_auc
    config, before = overrides['graph_lgbm'], guard()
    out = OUT/'runs'/name/cutoff
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'result.json').exists():
            r = json.loads((out/'result.json').read_text())
            assert r['augmentation_config'] == overrides
            return r
        mode = overrides.get('feature_mode', 'spectral_multi_pair')
        base, info, labels, x, v = arrays(cutoff, mode)
        x, names = matrix(x, info['feature_names'], config.get('slate_context'))
        v, vn = matrix(v, info['feature_names'], config.get('slate_context'))
        assert names == vn
        started = time.monotonic()
        train_offset, valid_offset = None, None
        if config.get('neural_offset'):
            shutil.copy2(OUT/'runs/spectral_multi_wide'/cutoff/'model.pt', out/'base_model.pt')
            train_offset, valid_offset = neural_logit(out/'base_model.pt', x), neural_logit(out/'base_model.pt', v)
        training = lgb.Dataset(x, label=labels['train'].ravel(), init_score=train_offset, params=params(config), free_raw_data=True)
        validation = lgb.Dataset(v, label=labels['valid'].ravel(), init_score=valid_offset, reference=training, free_raw_data=True)
        training.construct()
        validation.construct()
        del x
        gc.collect()
        model = fit_binned(config, training, validation, fixed_epochs)
        best = model.best_iteration or model.current_iteration()
        model.save_model(str(out/'model.txt'), num_iteration=best)
        if valid_offset is None:
            p = model.predict(v, num_iteration=best).reshape(-1, 10)
        else:
            from scipy.special import expit
            p = expit(model.predict(v, num_iteration=best, raw_score=True)+valid_offset).reshape(-1, 10)
        np.savez_compressed(out/'validation.npz', event_indices=labels['validation_indices'], labels=labels['valid'].reshape(-1, 10), predictions=p)
        iw = importance_weights(cutoff, labels['validation_indices'])[0]
        r = dict(name=name, cutoff=cutoff, method='graph_lightgbm', config=config, augmentation_config=overrides,
                 best_epoch=best, **arena_core.evaluate(labels['valid'], p), weighted_auc=weighted_auc(labels['valid'], p, iw),
                 feature_config=base['feature_config'], feature_dataset=base['feature_dataset'], feature_names=names,
                 parameters=sum(t['num_leaves'] for t in model.dump_model()['tree_info']), seconds=time.monotonic()-started,
                 library_version=lgb.__version__)
        assert all(sha(ROOT/f) == h for f, h in before.items() if f != 'competition/CURRENT_BEST.json')
        write_json(out/'result.json', r)
        print(json.dumps(r), flush=True)
        return r


def fit_final(name, final):
    from arena_ood_export import final_inputs
    r = json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
    out = final/'models'/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'recipe.json').exists():
        return out
    mode = r['augmentation_config'].get('feature_mode', 'spectral_multi_pair')
    base, raw, evaluation, ix, x, v, names = final_inputs(mode)
    x, names = matrix(x, names, r['config'].get('slate_context'))
    # Matrix returns a new names list only when adding context.
    original_names = names[:v.shape[1]]
    v, vn = matrix(v, original_names, r['config'].get('slate_context'))
    assert names == vn
    train_offset, valid_offset = None, None
    if r['config'].get('neural_offset'):
        shutil.copy2(ROOT/'competition/final_v6/models/spectral_multi_wide/model.pt', out/'base_model.pt')
        train_offset, valid_offset = neural_logit(out/'base_model.pt', x), neural_logit(out/'base_model.pt', v)
    training = lgb.Dataset(x, label=raw['labels'][ix].ravel(), init_score=train_offset, params=params(r['config']), free_raw_data=True)
    training.construct()
    del x
    gc.collect()
    model = fit_binned(r['config'], training, fixed_epochs=r['best_epoch'])
    model.save_model(str(out/'model.txt'))
    if valid_offset is None:
        p = model.predict(v).reshape(-1, 10)
    else:
        from scipy.special import expit
        p = expit(model.predict(v, raw_score=True)+valid_offset).reshape(-1, 10)
    np.save(out/'evaluation_predictions.npy', p)
    write_json(out/'recipe.json', dict(name=name, graph_tree=True, model_file='model.txt', backend='lightgbm',
               augmentation_config=r['augmentation_config'], feature_config=base['feature_config'], feature_names=names,
               model_config=r['config'], fixed_epochs=r['best_epoch'], training_events=len(ix), library_version=lgb.__version__))
    return out
