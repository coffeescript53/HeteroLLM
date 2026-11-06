#!/usr/bin/env python3
"""Read-only C3 analysis of frozen traces; outputs only in this milestone directory.

Completed same-device stream intervals are not kernel-only intervals. CPU interval
unions are used for elapsed CPU-call coverage; aggregate event calls cannot be
phase-attributed. Six T processes, not their 18 repetitions, are observations.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from heterollm.trace_analysis import analyze, strict_json, union_ns
from heterollm.timing_analysis import analyze_timing


def intersection_union(a, b):
    return union_ns([(max(x, p), min(y, q)) for x, y in a for p, q in b
                     if max(x, p) <= min(y, q)])


def intervals(spans):
    return [(s['begin_ns'], s['end_ns']) for s in spans]


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {'min': values[0], 'median': statistics.median(values),
            'p95_nearest_rank': values[max(0, (95 * len(values) + 99) // 100 - 1)],
            'max': values[-1]}


def phase_spans(spans, phase):
    return [s for s in spans if phase['begin'] <= s['begin_ns'] <= s['end_ns'] <= phase['end']]


def unique_graphs(records, backend_field):
    """This analysis requires one graph span per graph/backend in a phase."""
    indexed = {}
    for record in records:
        key = (record['graph'], record[backend_field])
        if key in indexed:
            raise ValueError('ambiguous graph/backend identity in phase: ' + str(key))
        indexed[key] = record
    return indexed


def summarize_phase(run, phase_id, phase, spans, devices, benchmark):
    spans = phase_spans(spans, phase)
    dev = [d for d in devices if d['phase'] == phase_id]
    by_kind = defaultdict(list)
    for s in spans:
        by_kind[s['kind']].append(s)
    stages = by_kind['copy.host_stage']
    if any((s['src'], s['dst']) != ('CUDA0', 'ROCm0') for s in stages):
        raise ValueError('unsupported host-stage direction; requires CUDA0 -> ROCm0')
    cross_fallback = [s for s in by_kind['copy.scheduler_fallback'] if s['src'] == 'CUDA0' and s['dst'] == 'ROCm0']
    waits = by_kind['backend.synchronize']
    alloc = by_kind['graph.allocate'] + by_kind['graph.reallocate'] + by_kind['graph.split_build']
    phases_ns = phase['end'] - phase['begin']
    result = {'run': run, 'phase_id': phase_id, 'phase': phase['name'], 'repetition': phase.get('repetition', 0),
              'phase_wall_ms': phases_ns / 1e6,
              'benchmark_elapsed_ms': benchmark / 1e6 if benchmark else None,
              'phase_to_benchmark_difference_ns': benchmark - phases_ns if benchmark else None,
              'cpu_thread_count': len({s['thread'] for s in spans}),
              'host_stage_count': len(stages), 'host_stage_logical_bytes': sum(s['logical_bytes'] for s in stages),
              'host_stage_cpu_union_ms': union_ns(intervals(stages)) / 1e6,
              'cross_fallback_count': len(cross_fallback),
              'cross_fallback_cpu_union_ms': union_ns(intervals(cross_fallback)) / 1e6,
              'backend_sync_cpu_union_ms': union_ns(intervals(waits)) / 1e6,
              'fallback_nested_sync_union_ms': intersection_union(intervals(cross_fallback), intervals(waits)) / 1e6,
              'fallback_nested_stage_union_ms': intersection_union(intervals(cross_fallback), intervals(stages)) / 1e6,
              'graph_split_allocate_count': len(alloc), 'graph_split_allocate_cpu_union_ms': union_ns(intervals(alloc)) / 1e6,
              'graph_cpu_submit_union_ms': union_ns(intervals(by_kind['graph.submit'])) / 1e6,
              'scheduler_async_accepted_count': len(by_kind['copy.scheduler_async_accepted'])}
    result['fallback_uncovered_cpu_ms'] = (union_ns(intervals(cross_fallback)) -
        intersection_union(intervals(cross_fallback), intervals(waits) + intervals(stages))) / 1e6
    result['cpu_kind_counts'] = dict(Counter(s['kind'] for s in spans))
    result['directions'] = dict(Counter(s['src'] + ' -> ' + s['dst'] for s in stages))
    result['graph_kinds'] = dict(Counter(d['backend'] + '/' + d['kind'] for d in dev if d['kind'].startswith('graph')))
    for backend in ('CUDA0', 'ROCm0'):
        gd = [d for d in dev if d['kind'].startswith('graph') and d['backend'] == backend]
        result[backend + '_graph_count'] = len(gd)
        result[backend + '_graph_stream_ms_sum'] = sum(d['elapsed_ms'] for d in gd)
        result[backend + '_backend_sync_cpu_union_ms'] = union_ns(intervals([s for s in waits if s['src'] == backend])) / 1e6
        result[backend + '_graph_enclosure_ms'] = quantiles([(d['observed'] - d['start_before']) / 1e6 for d in gd])
        result[backend + '_graph_enclosure_slack_ms'] = quantiles([(d['observed'] - d['start_before']) / 1e6 - d['elapsed_ms'] for d in gd])
    # Exact host-stage leg matching, independently after strict production replay.
    matched = []
    for s in by_kind['copy.host_stage.get'] + by_kind['copy.host_stage.set']:
        kind = 'copy_d2h' if s['kind'].endswith('get') else 'copy_h2d'
        backend = s['src'] if kind == 'copy_d2h' else s['dst']
        found = [d for d in dev if d['kind'] == kind and d['backend'] == backend and
                 all(d[k] == s[k] for k in ('thread', 'graph', 'split', 'logical_bytes')) and
                 s['begin_ns'] <= d['start_before'] <= d['observed'] <= s['end_ns']]
        if len(found) != 1:
            raise ValueError('ambiguous host stage native leg')
        matched.append(found[0])
    if len({d['id'] for d in matched}) != len(matched) or len(matched) != 2 * len(stages):
        raise ValueError('lost/duplicate native legs')
    result['host_stage_native_legs_count'] = len(matched)
    result['host_stage_native_legs_stream_ms_sum'] = sum(d['elapsed_ms'] for d in matched)
    result['all_native_copy_stream_ms_sum'] = sum(d['elapsed_ms'] for d in dev if d['kind'].startswith('copy'))
    # Common CPU-clock enclosure ordering only. No CUDA/HIP timestamp subtraction.
    graph_dev = unique_graphs((d for d in dev if d['kind'].startswith('graph')), 'backend')
    graph_cpu = unique_graphs((s for s in by_kind['graph.submit'] if s['src'] in ('CUDA0', 'ROCm0')), 'src')
    expected_graphs = {(g, b) for g, _ in graph_dev for b in ('CUDA0', 'ROCm0')}
    if set(graph_dev) != expected_graphs or set(graph_cpu) != expected_graphs:
        raise ValueError('graph identities do not match both CPU/device backends')
    if any((s['graph'], 'CUDA0') not in graph_dev for s in stages):
        raise ValueError('host-stage graph identity absent')
    links = []
    for s in stages:
        src, dst = graph_dev[(s['graph'], 'CUDA0')], graph_cpu[(s['graph'], 'ROCm0')]
        links.append(src['observed'] <= s['begin_ns'] <= s['end_ns'] <= dst['begin_ns'])
    result['source_completion_before_staging_before_destination_submit_count'] = sum(links)
    tg = sorted((d for d in dev if d['kind'].startswith('graph') and d['backend'] == 'CUDA0'), key=lambda d: d['start_before'])
    consecutive = [graph_dev[(a['graph'], 'ROCm0')]['observed'] <= b['start_before'] for a, b in zip(tg, tg[1:])]
    result['consecutive_graph_pairs'] = len(consecutive)
    result['prior_rocm_completion_before_next_cuda_enqueue_count'] = sum(consecutive)
    return result


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    runs = [(ROOT / 'runs/llama33-70b/stage-c2-phase-timing-diagnostic-20261027/01-equal-routing', 1)]
    runs += [(p, 3) for p in sorted((ROOT / 'runs/llama33-70b/stage-c2-overhead-20261027').glob('*-T'))]
    assert len(runs) == 7
    receipts, summaries, phases, originals = [], [], [], {}
    def read(path):
        raw = path.read_bytes()
        originals[path] = hashlib.sha256(raw).hexdigest()
        receipts.append({'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': originals[path]})
        return raw.decode()
    for directory, repetitions in runs:
        cpu, timing, stderr, stdout = [read(directory / n) for n in ('cpu-trace.jsonl', 'device-timing.jsonl', 'stderr.txt', 'stdout.txt')]
        csum = analyze(cpu, stderr)
        tsum = analyze_timing(timing, cpu, stderr, repetitions=repetitions)
        saved_cpu = json.loads(read(directory / 'trace-summary.json'))
        saved_timing = json.loads(read(directory / 'timing-summary.json'))
        if json.loads(json.dumps(csum)) != saved_cpu or json.loads(json.dumps(tsum)) != saved_timing:
            raise ValueError('strict replay differs from preserved summary: ' + str(directory))
        cb = [strict_json(s) for s in cpu.splitlines()]
        db = [strict_json(s) for s in timing.splitlines()]
        spans = [s for s in cb if s['type'] == 'span']
        devices = [d for d in db if d['type'] == 'device']
        bench = {('pp512' if b['n_prompt'] == 512 and b['n_gen'] == 0 else 'tg128'): b for b in json.loads(stdout)}
        assert set(bench) == {'pp512', 'tg128'}
        run = directory.parent.name + '/' + directory.name
        summaries.append({'run': run, 'repetitions': repetitions, 'cpu_detailed_spans': len(spans),
                          'cpu_aggregate_calls': csum['aggregate_calls'], 'native_completed_intervals': len(devices),
                          'strict_replay_matches_frozen_summaries': True, 'whole_process_event_aggregates': csum['aggregates']})
        for phase_id, phase in tsum['phases'].items():
            sample = bench[phase['name']]['samples_ns'][phase.get('repetition', 0)] if phase['name'] in bench else None
            phases.append(summarize_phase(run, phase_id, phase, spans, devices, sample))
    # Raw U/T benchmark timing samples: retain all 72 exactly; no historical substitution.
    samples = []
    for p in sorted((ROOT / 'runs/llama33-70b/stage-c2-overhead-20261027').glob('*-[UT]')):
        path = p / 'stdout.txt'
        text = read(path) if path not in originals else path.read_text()
        for b in json.loads(text):
            workload = 'pp512' if b['n_prompt'] == 512 else 'tg128'
            for repetition, ns in enumerate(b['samples_ns']):
                samples.append({'run': p.name, 'arm': p.name[-1], 'workload': workload, 'repetition': repetition,
                                'elapsed_ns': ns, 'tokens_per_second_from_ns': (512 if workload == 'pp512' else 128) * 1e9 / ns})
    assert len(samples) == 72
    # Hash the exact strict validators used, without altering production sources.
    for name in ('trace_analysis.py', 'timing_analysis.py'):
        read(ROOT / 'heterollm' / name)
    metrics = ['phase_wall_ms', 'host_stage_cpu_union_ms', 'host_stage_native_legs_stream_ms_sum',
               'cross_fallback_cpu_union_ms', 'fallback_nested_sync_union_ms', 'fallback_uncovered_cpu_ms',
               'backend_sync_cpu_union_ms', 'graph_split_allocate_cpu_union_ms', 'graph_cpu_submit_union_ms',
               'CUDA0_graph_stream_ms_sum', 'ROCm0_graph_stream_ms_sum', 'all_native_copy_stream_ms_sum']
    process = []
    for run in sorted({p['run'] for p in phases if 'overhead' in p['run']}):
        for name in ('pp512', 'tg128'):
            ps = [p for p in phases if p['run'] == run and p['phase'] == name]
            assert len(ps) == 3
            row = {'run': run, 'workload': name, 'internal_repetitions_correlated': 3}
            row.update({m: statistics.mean(p[m] for p in ps) for m in metrics})
            row.update({'host_stage_pct_of_phase': 100 * statistics.mean(p['host_stage_cpu_union_ms'] / p['phase_wall_ms'] for p in ps),
                        'native_stage_leg_pct_of_phase': 100 * statistics.mean(p['host_stage_native_legs_stream_ms_sum'] / p['phase_wall_ms'] for p in ps),
                        'graph_rocm_to_cuda_stream_duration_ratio': statistics.mean(p['ROCm0_graph_stream_ms_sum'] / p['CUDA0_graph_stream_ms_sum'] for p in ps)})
            process.append(row)
    stats = {}
    for name in ('pp512', 'tg128'):
        rows = [p for p in process if p['workload'] == name]
        stats[name] = {'independent_T_processes': 6, 'correlated_internal_repetitions_total': 18, 'metrics': {}}
        for metric in metrics + ['host_stage_pct_of_phase', 'native_stage_leg_pct_of_phase', 'graph_rocm_to_cuda_stream_duration_ratio']:
            vs = [r[metric] for r in rows]
            stats[name]['metrics'][metric] = {'process_mean': statistics.mean(vs), 'between_process_sample_sd': statistics.stdev(vs), 'process_min': min(vs), 'process_max': max(vs)}
    for path, digest in originals.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('input changed: ' + str(path))
    (OUT / 'ARCHIVED_TRACE_INPUT_HASHES.json').write_text(json.dumps({'inputs': receipts, 'input_files_unchanged_after': True}, indent=2) + '\n')
    (OUT / 'ARCHIVED_TRACE_ANALYSIS.json').write_text(json.dumps({'strict_replay': summaries, 'phases': phases,
        'six_process_statistics': stats, 'limitations': ['No cross-device clock subtraction; native intervals include device dependencies and gaps.',
        'CPU union coverage is not GPU idle time. Nested kinds are never added as independent costs.',
        'Host-stage logical bytes count parents once; not physical PCIe bytes.', 'Event aggregates have no phase timestamps.',
        'Six independent T launches, not eighteen repetitions, are statistical observations.',
        'A conservative CPU enqueue-to-completion enclosure is not an accurately aligned GPU activity timeline.']}, indent=2) + '\n')
    write_csv('ARCHIVED_PHASE_TABLE.csv', [{k: v for k, v in p.items() if not isinstance(v, (dict, list))} for p in phases])
    write_csv('ARCHIVED_PROCESS_TABLE.csv', process)
    write_csv('ORIGINAL_72_BENCHMARK_SAMPLES.csv', samples)
    print(json.dumps({'strict_trace_pairs': len(runs), 'phases': len(phases), 'original_timing_samples': len(samples), 'input_files': len(originals), 'all_inputs_unchanged': True,
                      'six_process_statistics': stats}, indent=2))


if __name__ == '__main__':
    main()
