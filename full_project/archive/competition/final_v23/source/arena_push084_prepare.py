"""Wait for the complete campaign, then audit the frozen recipe sequentially."""
import fcntl
import json
import os
import shutil
import subprocess
import time
from arena_ood import ROOT, guard, sha, write_json
from arena_push084 import OUT, command

EXPECTED = {'coclick_cross', 'coclick_wide', 'exposure_cross', 'exposure_wide',
            'daily_coclick_cross', 'daily_coclick_wide', 'rich_coclick_wide', 'balanced_coclick_wide',
            'rank_coclick_wide', 'spectral_wide', 'softmax30_wide', 'softmax90_wide',
            'gate_coclick_wide', 'gate_rich_cross', 'joint_graph_wide', 'weekend_wide',
            'spectral64_wide', 'spectral_multi_wide', 'normalized_spectral_wide',
            'origin_coclick_wide', 'origin_spectral_wide', 'bias80_wide', 'bias240_wide'}


def clear_completed_feature_cache():
    if shutil.disk_usage(ROOT).free >= 3*1024**3:
        return
    files = [f for p in (ROOT/'competition/ood/features').iterdir()
             if p.name.startswith(('2025-01-25_', '2025-01-26_')) for f in p.glob('*.npy')]
    rows = [dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=sha(f)) for f in files]
    write_json(OUT/'post_training_cache_cleanup.json', dict(files=rows, bytes=sum(r['bytes'] for r in rows),
               policy='Only completed, regenerable fitting/validation feature matrices removed. All model checkpoints, frozen representations, validation predictions, indices and feature metadata retained. Raw data and canonical feature caches untouched.'))
    for f in files:
        f.unlink()
    print(json.dumps(dict(stage='cache_reclaimed', gib=sum(r['bytes'] for r in rows)/1024**3)), flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    with (OUT/'.prepare.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            with (OUT/'.campaign.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = OUT/'selection.json'
                selection = json.loads(path.read_text()) if path.exists() else {}
                if {r['name'] for r in selection.get('candidates', [])} == EXPECTED:
                    before = guard()
                    assert before == json.loads((OUT/'incumbent.json').read_text())
                    clear_completed_feature_cache()
                    command(['arena_push084_reblend.py'], 'final_weight_selection')
                    command(['arena_push084_audit.py'], 'final_two_origin_audits')
                    command(['arena_push084_audit.py', '--third'], 'frozen_third_origin_audit')
                    assert guard() == before
                    rows = {day: json.loads((OUT/('audit_'+day+'.json')).read_text()) for day in ['26', '25', '24']}
                    write_json(OUT/'prepared.json', dict(stage='audits_completed', selection_sha256=sha(path), audits=rows,
                                                       incumbent_unchanged=True, prepared_at=time.time()))
                    write_json(OUT/'state.json', dict(stage='ready_for_export', pid=os.getpid(), updated=time.time()))
                    print(json.dumps(dict(stage='ready_for_export', metrics={d: r['new_auc'] for d, r in rows.items()})), flush=True)
                    return
            time.sleep(15)


if __name__ == '__main__':
    run()
