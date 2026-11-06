"""Recalculate throughput from integer durations; launches are independent units."""
import math
import random
import statistics as st
from .config import digest


def describe(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('throughput samples must be finite and positive')
    mean = st.mean(values)
    return {'n':len(values),'mean':mean,'median':st.median(values),
            'sample_sd':st.stdev(values) if len(values)>1 else None,
            'cv':st.stdev(values)/mean if len(values)>1 else None,
            'min':min(values),'max':max(values)}


def parse_benchmark(rows, config, reps, *, load_mode='mmap'):
    if load_mode not in ('mmap', 'none'):
        raise ValueError('unsupported benchmark loading contract')
    if not isinstance(rows,list) or len(rows) != 2:
        raise ValueError('expected exactly separate PP and TG results')
    results = {}
    expected = {(config.prompt_tokens,0):'pp',(0,config.generation_tokens):'tg'}
    for row in rows:
        key = expected.get((row.get('n_prompt'),row.get('n_gen')))
        if key is None or key in results or row.get('n_depth') != config.depth:
            raise ValueError('benchmark workload mismatch')
        checks = {'n_batch':config.batch,'n_ubatch':config.ubatch,'n_threads':config.threads,
                  'type_k':config.cache_k,'type_v':config.cache_v,'flash_attn':0,'fit_target':0,
                  'split_mode':'layer','load_mode':load_mode,'lazy_mode':'off','fit_min_ctx':0,
                  'no_kv_offload':False,'no_op_offload':0,'no_host':False,'repack':True,
                  'embeddings':False,'n_cpu_moe':0,'tensor_buft_overrides':'none'}
        for field, expected_value in checks.items():
            if row.get(field) != expected_value:
                raise ValueError(f'benchmark {field} mismatch: {row.get(field)!r}')
        times = row.get('samples_ns')
        if not isinstance(times,list) or len(times) != reps or any(type(t) is not int or t <= 0 for t in times):
            raise ValueError('missing, invalid, or incomplete duration samples')
        n = row['n_prompt'] + row['n_gen']
        samples = [n*1e9/t for t in times]
        results[key] = {'samples_ns':times,'samples_tokens_s':samples,**describe(samples)}
    return results


def aggregate(records, phase='final'):
    groups = {}
    signatures = {r['comparison_key'] for r in records if r.get('status')=='ok' and r.get('phase')==phase}
    if len(signatures)>1:
        raise ValueError('model/runtime/workload/hardware differ; comparison refused')
    seen = set()
    identities = {}
    for r in records:
        if r.get('status') != 'ok' or r.get('phase') != phase:
            continue
        if r['run_id'] in seen:
            raise ValueError('duplicate independent launch')
        seen.add(r['run_id'])
        if not r.get('validation_key'):
            raise ValueError('unvalidated benchmark cannot be ranked')
        name = r['policy']['name']
        identity = digest({'policy':r['policy'],'validation_key':r['validation_key']})
        if name in identities and identities[name] != identity:
            raise ValueError('same policy name refers to different policies or validation identities')
        identities[name] = identity
        g = groups.setdefault(r['policy']['name'],{'pp':[],'tg':[],'run_ids':[],'rounds':[]})
        for metric in ('pp','tg'):
            g[metric].append(r['metrics'][metric]['mean'])
        g['run_ids'].append(r['run_id'])
        g['rounds'].append(r.get('round'))
    return {name:{'pp':describe(g['pp']),'tg':describe(g['tg']), 'launch_means':{'pp':g['pp'],'tg':g['tg']},
                  'run_ids':g['run_ids'],'rounds':g['rounds']} for name,g in groups.items()}


def finalists(records):
    scores = aggregate(records,'explore')
    if not scores:
        return []
    # Remeasure all valid baseline classes and the best explored nonbaseline.
    baseline = {'cuda-only','rocm-only','mixed-default'} & scores.keys()
    others = set(scores)-baseline
    if others:
        baseline.add(max(sorted(others),key=lambda n:scores[n]['tg']['mean']))
    return sorted(baseline)


def interval_ratio(selected, baseline):
    if len(selected)<3 or len(baseline)<3:
        return None
    rng = random.Random(0)
    ratios = sorted(st.mean(rng.choices(selected,k=len(selected)))/st.mean(rng.choices(baseline,k=len(baseline))) for _ in range(4000))
    return [ratios[100],ratios[3899]]


def selection(records, required_launches=3):
    groups = aggregate(records)
    eligible = {n:g for n,g in groups.items() if g['tg']['n']>=required_launches}
    if not eligible:
        return {'selected':None,'groups':groups,'reason':'no valid policy has enough independent final launches'}
    selected = max(sorted(eligible),key=lambda n:eligible[n]['tg']['mean'])
    baselines = {n:g for n,g in eligible.items() if n in ('cuda-only','rocm-only','mixed-default')}
    strongest = max(sorted(baselines),key=lambda n:baselines[n]['tg']['mean']) if baselines else None
    comparisons = {}
    for name in sorted({'mixed-default',strongest}-{None}):
        if name in eligible:
            a,b = eligible[selected],eligible[name]
            interval = [1.0,1.0] if name==selected else interval_ratio(a['launch_means']['tg'],b['launch_means']['tg'])
            comparisons[name] = {'speedup':a['tg']['mean']/b['tg']['mean'], 'bootstrap_95_interval':interval,
                                 'reliable_improvement_claim':False,
                                 'note':'descriptive interval only: small n, winner selection, and time drift limit inference'}
    return {'selected':selected,'strongest_baseline':strongest,'objective':'mean independent-launch TG tokens/s',
            'groups':groups,'comparisons':comparisons,'claim':'fastest valid observed policy among tested finalists; no global-optimum claim'}
