"""Train the explicitly indexed 619-motion PHP collection for 100000 update groups."""
import argparse
from datetime import datetime
from pathlib import Path
import json
import csv
import subprocess
import sys
import shutil
import hashlib
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blinkmimic.php.recipe import UPSTREAM_MAIN

SOURCE='/home/ljc/php_parkour/experiments/supermimic_parkour_isaacsim/data/php_full17_merged619_20261104'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def freeze_inputs(root,run,catalog):
    """Keep this long run independent of later edits to the PHP catalog/code."""
    data=run/'dataset'
    source=run/'source'
    if data.exists() or source.exists():
        raise FileExistsError('Snapshots already exist; use --resume with this run or choose a new log directory')
    staging=run/'dataset.tmp'
    staging.mkdir()
    for cohort in catalog['cohorts']:
        shutil.copytree(root/cohort,staging/cohort)
    shutil.copy2(root/'collection_manifest.json',staging/'collection_manifest.json')
    for path in root.glob('README*'):
        if path.is_file(): shutil.copy2(path,staging/path.name)
    for row in catalog['active']:
        for key,sha in [('path','sha256'),('terrain','terrain_sha256')]:
            path=staging/row[key]
            if not path.resolve().is_relative_to(staging.resolve()) or digest(path)!=row[sha]:
                raise ValueError(f'Snapshot checksum/path mismatch: {row[key]}')
    staging.rename(data)
    checkout=Path(__file__).resolve().parents[1]
    source.mkdir()
    for name in ['blinkmimic','supermimic_parkour','scripts']:
        shutil.copytree(checkout/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['README.md','NOTICE','pyproject.toml']:
        shutil.copy2(checkout/name,source/name)
    files={str(path.relative_to(source)):digest(path) for path in sorted(source.rglob('*')) if path.is_file()}
    (run/'input_snapshots.json').write_text(json.dumps({'original_dataset':str(root),
        'dataset_snapshot':str(data),'collection_sha256':digest(data/'collection_manifest.json'),
        'verified_active_files':2*len(catalog['active']),'source_files_sha256':files},indent=2))
    print(f'Frozen {len(catalog["active"])} motions and training source under {run}',flush=True)
    return data,source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',default=SOURCE)
    parser.add_argument('--log_dir',default=None)
    parser.add_argument('--resume',default=None)
    parser.add_argument('--resume-source',type=Path,help='use a new frozen code snapshot for a resumed run')
    parser.add_argument('--no-final-eval',action='store_true')
    args=parser.parse_args()
    if args.resume_source and not args.resume:
        parser.error('--resume-source requires --resume')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run=Path(args.log_dir or f'logs/blinkmimic_php619_100k_{stamp}').resolve()
    run.mkdir(parents=True,exist_ok=True)
    if args.resume and (run/'launch.json').exists():
        previous=json.loads((run/'launch.json').read_text())
        root=Path(previous['dataset_snapshot'])
        resume_record=run/'resume_launch.json'
        source_record=json.loads(resume_record.read_text()) if resume_record.exists() else previous
        source=(args.resume_source or Path(source_record['source_snapshot'])).resolve()
        if not (source/'blinkmimic/train.py').is_file():
            parser.error('The resume source must contain blinkmimic/train.py')
    else:
        root=Path(args.dataset_dir).expanduser().resolve()
        source=None
    catalog=json.loads((root/'collection_manifest.json').read_text())
    if catalog['active_clips']!=619 or len(catalog['active'])!=619:
        parser.error('Expected the explicit 619-active-motion collection')
    if source is None:
        if args.resume:
            parser.error('Resume requires this run\'s launch.json and frozen inputs')
        root,source=freeze_inputs(root,run,catalog)
    overrides=['--robot','php','--motion_dir',str(root),'--num_envs','2048',
        '--buffer_steps','128','--pi_microbatch','128','--eprobe_batch','32','--eprobe_async','0',
        '--iterations','100000','--lr_decay_iters','100000','--eprobe_start','1000','--eprobe_every','1000',
        '--ckpt_every','1000','--recovery_every','1000','--log_every','25','--log_dir',str(run)]
    if args.resume: overrides+=['--resume',str(Path(args.resume).resolve())]
    command=[sys.executable,'-u','-m','blinkmimic.train',*UPSTREAM_MAIN,*overrides]
    evaluation=[sys.executable,'scripts/evaluate_php_supermimic.py','--checkpoint',str(run/'ckpt_100000.pt'),
                '--dataset-dir',str(root),'--output',str(run/'final_evaluation_variant0.json')]
    if args.resume:
        command=previous['training_command']+['--resume',str(Path(args.resume).resolve())]
        evaluation=previous['final_evaluation_command']
    record=run/('resume_launch.json' if args.resume else 'launch.json')
    record.write_text(json.dumps({'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'training_command':command,'final_evaluation_command':evaluation,'final_evaluation_enabled':not args.no_final_eval,
        'dataset_snapshot':str(root),'source_snapshot':str(source),
        'dataset_active_clips':619,'quarantined_clips_excluded':catalog['quarantine_count'],
        'cohorts':catalog['cohorts'],'from_scratch':args.resume is None,'iteration_unit':'update_group'},indent=2))
    subprocess.run(command,check=True,cwd=source)
    rows=list(csv.DictReader((run/'train_log.csv').open()))
    complete=bool(rows) and int(rows[-1]['iteration'])==100000
    if not args.no_final_eval and complete:
        subprocess.run(evaluation,check=True,cwd=source)
    elif not complete:
        print('Training ended before the target; recovery state retained, final evaluation skipped.',flush=True)


if __name__=='__main__': main()
