"""Offline paired observer-effect and deterministic full-logit acceptance."""
from array import array
import math
from pathlib import Path
import re
import statistics
import sys
from .stage_c import strict_json
from .hardware import sha256_file

ORDERS=('UT','TU','UT','TU','UT','TU')
ATOL=1e-4
RTOL=1e-5
VOCAB=128256
ROWS=129

def paired_summary(observations):
    """Six paired process means; never inflate n using internal repetitions."""
    expected=[(block,arm) for block,order in enumerate(ORDERS,1) for arm in order]
    if any(type(r.get('block')) is not int or type(r.get('arm')) is not str for r in observations) or [(r.get('block'),r.get('arm')) for r in observations]!=expected:
        raise ValueError('requires exactly six ordered complete independent pairs')
    ids=[r.get('run_id') for r in observations]
    if any(type(x)is not str or not x for x in ids) or len(set(ids))!=12:
        raise ValueError('duplicate/missing independent process identity')
    if len({r.get('contract_sha256') for r in observations})!=1 or not re.fullmatch('[0-9a-f]{64}',observations[0].get('contract_sha256','')):
        raise ValueError('mixed execution contracts')
    normalized=[]
    for row in observations:
        if row.get('status')!='passed' or set(row.get('samples_ns',{}))!={'pp512','tg128'}:
            raise ValueError('incomplete/failed workload')
        means={};rates={}
        for phase,n in [('pp512',512),('tg128',128)]:
            values=row['samples_ns'][phase]
            if type(values)is not list or len(values)!=3 or any(type(v)is not int or v<=0 for v in values):
                raise ValueError('three positive integer repetition durations required')
            means[phase]=statistics.mean(values)
            rates[phase]={'samples_tokens_sec':[n*1e9/v for v in values],
                          'mean_tokens_sec':statistics.mean(n*1e9/v for v in values)}
        normalized.append({**row,'mean_ns':means,'throughput':rates})
    result={}
    for phase in ('pp512','tg128'):
        pairs=[]
        for block,order in enumerate(ORDERS,1):
            members={r['arm']:r for r in normalized if r['block']==block}
            ratio=members['T']['mean_ns'][phase]/members['U']['mean_ns'][phase]
            pairs.append({'block':block,'order':order,'U_mean_ns':members['U']['mean_ns'][phase],
                          'T_mean_ns':members['T']['mean_ns'][phase], 'elapsed_difference_ns':members['T']['mean_ns'][phase]-members['U']['mean_ns'][phase],
                          'log_elapsed_ratio':math.log(ratio),'overhead_percent':100*(ratio-1)})
        logs=[p['log_elapsed_ratio'] for p in pairs];mean=statistics.mean(logs);sd=statistics.stdev(logs)
        half=2.570581835636305*sd/math.sqrt(6) # Student t, df=5, nominal two-sided 95%
        result[phase]={'pairs':pairs,'independent_pairs':6,'independent_processes_per_arm':6,
            'mean_log_elapsed_ratio':mean,'sd_log_elapsed_ratio':sd,
            'geometric_overhead_percent':100*math.expm1(mean),
            'nominal_95pct_ci_percent':[100*math.expm1(mean-half),100*math.expm1(mean+half)],
            'arithmetic_mean_pair_percent':statistics.mean(p['overhead_percent'] for p in pairs),
            'sd_pair_percent':statistics.stdev(p['overhead_percent'] for p in pairs),
            'by_order':{order:{'n':3,'mean_log_ratio':statistics.mean(p['log_elapsed_ratio'] for p in pairs if p['order']==order)} for order in ('UT','TU')},
            'arm_process_means':{arm:{'mean_elapsed_ns':statistics.mean(r['mean_ns'][phase] for r in normalized if r['arm']==arm),
                'sd_elapsed_ns':statistics.stdev(r['mean_ns'][phase] for r in normalized if r['arm']==arm),
                'mean_tokens_sec':statistics.mean(r['throughput'][phase]['mean_tokens_sec'] for r in normalized if r['arm']==arm),
                'sd_tokens_sec':statistics.stdev(r['throughput'][phase]['mean_tokens_sec'] for r in normalized if r['arm']==arm)} for arm in ('U','T')}}
    return {'schema':'heterollm.overhead.summary.v1','observations':normalized,'workloads':result,
        'uncertainty':'Nominal t interval assumes independent approximately normal paired log ratios. Six blocks are small; time trends/serial dependence and multiple endpoints are not resolved. Report order strata and all pairs; no automatic causal/bottleneck claim.',
        'unmeasured':['kernel_only_execution','precise_cross_vendor_overlap','U_phase_markers','U_trace_routing','isolated_loading_time','isolated_flush_time']}

def capture_metadata(directory,arm):
    directory=Path(directory);meta=directory/'predictions.jsonl';raw=directory/'logits.f32'
    if meta.is_symlink() or raw.is_symlink() or meta.stat().st_size>2*1024**2 or raw.stat().st_size!=ROWS*VOCAB*4:
        raise ValueError('capture missing/size/symlink mismatch')
    records=[strict_json(line) for line in meta.read_text().splitlines()]
    header={'type':'header','schema':'heterollm.equivalence.v1','arm':arm,'vocabulary':VOCAB,'rows':ROWS,'dtype':'little-endian-float32'}
    if len(records)!=ROWS+2 or records[0]!=header or records[-1]!={'type':'footer','rows':ROWS,'complete':True} or records[-1].get('complete') is not True or type(records[0].get('rows')) is not int or type(records[0].get('vocabulary')) is not int or type(records[-1].get('rows')) is not int:
        raise ValueError('incomplete capture header/footer')
    for i,row in enumerate(records[1:-1]):
        if set(row)!={'type','row','phase','position','argmax','piece_hex'} or row['type']!='row' or type(row['row'])is not int or row['row']!=i or row['phase']!=('pp512' if i==0 else 'tg128') or type(row['position'])is not int or row['position']!=(511 if i==0 else i-1) or type(row['argmax'])is not int or not 0<=row['argmax']<VOCAB or type(row['piece_hex'])is not str or not re.fullmatch('(?:[0-9a-f]{2}){0,4096}',row['piece_hex']):
            raise ValueError('capture row metadata')
    return records[1:-1]

def read_logits(handle):
    values=array('f');values.frombytes(handle.read(VOCAB*4))
    if sys.byteorder!='little':values.byteswap()
    if len(values)!=VOCAB or any(not math.isfinite(v) for v in values):raise ValueError('truncated/nonfinite logits')
    return values

def validate_capture(directory,arm):
    rows=capture_metadata(directory,arm)
    with (Path(directory)/'logits.f32').open('rb') as f:
        for row in rows:
            values=read_logits(f)
            if max(range(VOCAB),key=values.__getitem__)!=row['argmax']:raise ValueError('incorrect recorded argmax')
    return {'schema':'heterollm.equivalence.capture.v1','complete':True,'arm':arm,'rows':ROWS,'vocabulary':VOCAB,
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in (Path(directory)/'logits.f32',Path(directory)/'predictions.jsonl')}}

def compare_captures(u,t):
    ur=capture_metadata(u,'U');tr=capture_metadata(t,'T');details=[];total_bad=0;total_sq=0.;maximum=0.;different=0
    with (Path(u)/'logits.f32').open('rb') as uf,(Path(t)/'logits.f32').open('rb') as tf:
        for a,b in zip(ur,tr):
            uv=read_logits(uf);tv=read_logits(tf)
            if max(range(VOCAB),key=uv.__getitem__)!=a['argmax'] or max(range(VOCAB),key=tv.__getitem__)!=b['argmax']:
                raise ValueError('incorrect recorded argmax')
            max_abs=0.;bad=0;sq=0.;max_scaled=0.;diff=0
            for x,y in zip(uv,tv):
                delta=abs(x-y);limit=ATOL+RTOL*abs(x)
                bad+=delta>limit;diff+=x!=y;sq+=delta*delta
                max_abs=max(max_abs,delta);max_scaled=max(max_scaled,delta/limit)
            predictions_match=(a['argmax'],a['piece_hex'])==(b['argmax'],b['piece_hex'])
            details.append({'row':a['row'],'phase':a['phase'],'position':a['position'],'maximum_absolute_difference':max_abs,
                'maximum_fraction_of_tolerance':max_scaled,'rmse':math.sqrt(sq/VOCAB),'outside_tolerance':bad,
                'different_values':diff,'predictions_match':predictions_match})
            total_bad+=bad;total_sq+=sq;different+=diff;maximum=max(maximum,max_abs)
    return {'schema':'heterollm.equivalence.comparison.v1','passed':total_bad==0 and all(r['predictions_match'] for r in details),
        'atol':ATOL,'rtol':RTOL,'reference':'U','compared_values':ROWS*VOCAB,'outside_tolerance':total_bad,
        'different_values':different,'maximum_absolute_difference':maximum,'rmse':math.sqrt(total_sq/(ROWS*VOCAB)),
        'positions':details,'scope':'One fixed teacher-forced fixture; exact greedy predictions and piece bytes. Not free-running generation or general numerical equivalence.'}
