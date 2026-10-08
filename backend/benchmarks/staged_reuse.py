"""Actual stage-and-confirm path: bounded copy versus owned-snapshot reuse."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from ingestion_batching import exact_typed_fingerprint, gold_specs
from ingestion_parallelism import spec_path


def worker(spec, mode, output):
    # Ensure this benchmark never starts provider requests.
    os.environ.update(OPENAI_API_KEY='', AI_GATEWAY_API_KEY='', COLUMN_INTERPRETER='off',
                      ANALYTICO_NATIVE_VALIDATION='0', ANALYTICO_DUCKDB_THREADS='4')
    from services import import_preview as preview
    from storage import DATASETS
    from modules.import_policy import ImportSettings
    ds = None
    item = None
    experiments = ExitStack()
    try:
        if mode != 'production':
            from ingestion_batching import install_experiment
            install_experiment('baseline', 4)
        if mode == 'optimized':
            from parallel_candidates import install
            from parser_candidates import install as parser_install
            install('parallel_stats')
            experiments.enter_context(parser_install('overlap_validation'))
        started = time.perf_counter()
        staged = preview.stage_import(spec_path(spec), spec['filename'],
                                      ImportSettings(delimiter=spec['delimiter']))
        staged_seconds = time.perf_counter() - started
        item = preview._imports[staged['import_id']]
        stage_inode = item.path.stat().st_ino
        stage_path = item.path
        confirming = time.perf_counter()
        if mode == 'copy':
            with patch('utils.source_files.os.link', side_effect=OSError('benchmark copy baseline')):
                response = preview.confirm_import(item.id, item.settings, ai_column_analysis=False)
        else:
            response = preview.confirm_import(item.id, item.settings, ai_column_analysis=False)
        finished = time.perf_counter()
        ds = DATASETS[response.dataset_id]
        source_path = ds.disk.source_path if hasattr(ds, 'disk') else ds.source_path
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {'dataset': spec['id'], 'mode': mode, 'status': 'ok',
                  'stage_seconds': staged_seconds, 'confirm_seconds': finished-confirming,
                  'ready_seconds': finished-started, 'rows': ds.row_count,
                  'stage_removed': not stage_path.exists(),
                  'source_reused': source_path.stat().st_ino == stage_inode,
                  'peak_rss_bytes': peak if sys.platform=='darwin' else peak*1024}
        if hasattr(ds,'disk'):
            record['phases'] = ds.disk.ingestion_timings
            ds.disk._connection.execute("SET threads=1")
            ds.disk._connection.execute("SET memory_limit='1GB'")
            record['fingerprint'] = exact_typed_fingerprint(ds.disk)
        else:
            import pandas as pd
            record['fingerprint']={'values':str(int(pd.util.hash_pandas_object(ds.df,index=True).sum())),
                'dtypes':{k:str(v) for k,v in ds.df.dtypes.items()},'schema':ds.column_schema,
                'roles':ds.column_types,'missing':ds.raw_missing_counts,'rows':ds.row_count}
        with source_path.open('rb') as stream:
            record['source_matches'] = hashlib.file_digest(stream,'sha256').hexdigest()==spec['sha256']
        record['rows_match']=ds.row_count==spec['rows']
        output.write_text(json.dumps(record,default=str))
    finally:
        experiments.close()
        if ds is not None:
            DATASETS.pop(ds.id,None);ds.close()
        if item is not None and item.id in preview._imports:
            preview.cancel_import(item.id)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker');p.add_argument('--mode',choices=['copy','reuse','optimized','production'])
    p.add_argument('--modes',nargs='+',choices=['copy','reuse','optimized','production'],default=['copy','reuse','optimized'])
    p.add_argument('--output',type=Path)
    p.add_argument('--datasets',nargs='+',default=['taxi','retail','wine'])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-root',type=Path,default=Path('/private/tmp/analytico-staged-reuse-2026-10-07'))
    a=p.parse_args();specs=gold_specs()
    if a.worker:return worker(next(s for s in specs if s['id']==a.worker),a.mode,a.output)
    specs=[s for s in specs if s['id'] in a.datasets]
    a.output_root.mkdir(parents=True,exist_ok=True)
    jobs=[(s,m,r) for s in specs for m in a.modes for r in range(a.repeats)]
    random.Random(20261007).shuffle(jobs);records=[]
    for spec,mode,repeat in jobs:
        output=a.output_root/f"{spec['id']}-{mode}-{repeat}.json"
        output.unlink(missing_ok=True)
        with (a.output_root/'worker.log').open('a') as log:
            run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',spec['id'],
                '--mode',mode,'--output',str(output)],stdout=log,stderr=log)
        record=json.loads(output.read_text()) if output.exists() else {'dataset':spec['id'],'mode':mode,'status':'crash','exit_code':run.returncode}
        record['repeat']=repeat;records.append(record)
        (a.output_root/'records.json').write_text(json.dumps(records,default=str))
        print(json.dumps({k:record.get(k) for k in ['dataset','mode','status','ready_seconds','source_reused']}),flush=True)
    summary=[]
    for spec in specs:
        baseline=next(r['fingerprint'] for r in records if r['dataset']==spec['id'] and r['status']=='ok')
        for mode in a.modes:
            rs=[r for r in records if r['dataset']==spec['id'] and r['mode']==mode]
            valid=[r for r in rs if r['status']=='ok']
            summary.append({'dataset':spec['id'],'mode':mode,'runs':len(rs),'successes':len(valid),
                'medians':{k:statistics.median(r[k] for r in valid) for k in ['stage_seconds','confirm_seconds','ready_seconds','peak_rss_bytes']},
                'fingerprint_matches':sum(r.get('fingerprint')==baseline for r in rs),
                'source_matches':sum(r.get('source_matches',False) for r in rs),
                'row_matches':sum(r.get('rows_match',False) for r in rs),
                'stage_cleanup_checks':sum(r.get('stage_removed',False) for r in rs),
                'reused_runs':sum(r.get('source_reused',False) for r in rs)})
    (a.output_root/'summary.json').write_text(json.dumps({'ai_calls':0,'timing_scope':'Backend staging + confirmation; excludes transfer/render/AI; verification excluded','summary':summary},indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
