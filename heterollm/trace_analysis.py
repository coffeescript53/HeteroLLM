"""Strict CPU-only trace acceptance; no GPU/phase attribution is inferred."""
from collections import defaultdict
import json

AGGREGATE_KINDS = ['event.record_submit', 'event.synchronize']
DETAIL_KINDS = {
    'backend.synchronize', 'graph.submit', 'event.wait_submit',
    'copy.buffer_interface_attempt', 'copy.buffer_interface_accepted',
    'copy.host_to_backend', 'copy.backend_to_host', 'copy.host_stage',
    'copy.host_stage.get', 'copy.host_stage.set', 'copy.public_async_fallback',
    'graph.split_build', 'graph.allocate', 'graph.reallocate', 'graph.compute_splits',
    'split.cpu_scope', 'copy.scheduler_async_attempt', 'copy.scheduler_fallback',
    'copy.scheduler_async_accepted',
}
UINT64_MAX = 2**64 - 1


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate field: ' + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('nonfinite value: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def union_ns(intervals):
    """Elapsed union, not a sum that double-counts nested/concurrent intervals."""
    end = None
    total = 0
    for begin, stop in sorted(intervals):
        if not integer(begin) or not integer(stop) or stop < begin:
            raise ValueError('invalid interval')
        total += max(0, stop - max(begin, end if end is not None else begin))
        end = max(stop, end if end is not None else stop)
    return total


def analyze_v1(text, stderr, *, allow_empty=False):
    if 'HETEROLLM_TRACE_ERROR' in stderr:
        raise ValueError('recorder reported an error')
    if len(text.encode()) > 64 * 1024**2:
        raise ValueError('trace exceeds offline 64 MiB cap')
    rows = [strict_json(line) for line in text.splitlines()]
    if len(rows) < (2 if allow_empty else 3) or any(type(r) is not dict for r in rows):
        raise ValueError('missing trace records')
    header, *events, footer = rows
    if (set(header) != {'type','schema','clock','time_domain','phase','capacity','record_bytes'}
        or header['type'] != 'header' or header['schema'] != 'heterollm.cpu_trace.v1'
        or header['clock'] != 'steady_clock_ns' or header['time_domain'] != 'cpu'
        or header['phase'] != 'unknown' or not integer(header['capacity'], 1)
        or header['capacity'] > 65536 or not integer(header['record_bytes'], 1)):
        raise ValueError('unsupported header or timing domain')
    if (set(footer) != {'type','recorded','dropped','unfinished','metadata_truncated','complete'}
        or footer['type'] != 'footer' or footer['complete'] is not True
        or any(not integer(footer[k]) for k in ('recorded','dropped','unfinished','metadata_truncated'))
        or any(footer[k] for k in ('dropped','unfinished','metadata_truncated'))
        or footer['recorded'] != len(events) or len(events) > header['capacity']):
        raise ValueError('incomplete or overflowed trace')
    kinds = defaultdict(lambda: {'count':0, 'inclusive_cpu_ns':0})
    directions = defaultdict(lambda: {'copies':0, 'logical_bytes':0, 'inclusive_cpu_ns':0})
    fields = {'type','seq','thread','begin_ns','end_ns','graph','split','copy','n_copies','logical_bytes','kind','src','dst'}
    waits = []
    for seq, e in enumerate(events):
        if (set(e) != fields or e['type'] != 'span'
            or any(not integer(e[k]) for k in ('seq','thread','begin_ns','end_ns','graph','n_copies','logical_bytes'))
            or any(not integer(e[k], -1) for k in ('split','copy'))
            or e['seq'] != seq or e['end_ns'] < e['begin_ns'] or e['end_ns'] == 0
            or any(type(e[k]) is not str for k in ('kind','src','dst'))
            or not e['kind']):
            raise ValueError('invalid span')
        elapsed = e['end_ns'] - e['begin_ns']
        kinds[e['kind']]['count'] += 1
        kinds[e['kind']]['inclusive_cpu_ns'] += elapsed
        # Count only the parent staged-copy record, never nested get/set legs.
        if e['kind'] == 'copy.host_stage':
            d = directions[e['src'] + ' -> ' + e['dst']]
            d['copies'] += 1; d['logical_bytes'] += e['logical_bytes']; d['inclusive_cpu_ns'] += elapsed
        if e['kind'] in ('backend.synchronize','event.synchronize'):
            waits.append((e['begin_ns'],e['end_ns']))
    return {'schema':'heterollm.cpu_trace.summary.v1', 'complete':True,
            'phase':'unknown', 'gpu_execution_time_ns':None, 'gpu_overlap_ns':None,
            'cpu_wait_union_ns':union_ns(waits), 'kinds':dict(kinds),
            'host_staged_directions':dict(directions), 'events':len(events),
            'limitations':['CPU spans only; no completed GPU activity or phase attribution',
                           'Inclusive kind totals overlap; do not sum them as wall time',
                           'Logical tensor bytes are not PCIe bus traffic',
                           'Trace completeness does not prove model correctness or safe cleanup']}


def analyze(text, stderr):
    """Keep historical v1 validation; v2 distinguishes exact aggregation from loss."""
    if 'HETEROLLM_TRACE_ERROR' in stderr:
        raise ValueError('recorder reported an error')
    if len(text.encode()) > 64 * 1024**2:
        raise ValueError('trace exceeds offline 64 MiB cap')
    rows = [strict_json(line) for line in text.splitlines()]
    if len(rows) < 2 or any(type(r) is not dict for r in rows):
        raise ValueError('missing trace records')
    if rows[0].get('schema') == 'heterollm.cpu_trace.v1':
        return analyze_v1(text, stderr)
    h, *records, f = rows
    old_header = {'type','schema','clock','time_domain','phase','capacity','record_bytes'}
    if (set(h) != old_header | {'aggregate_capacity','aggregate_record_bytes','histogram','detail_policy','aggregate_kinds'}
        or h['schema'] != 'heterollm.cpu_trace.v2'
        or h['aggregate_capacity'] != 32 or type(h['aggregate_capacity']) is not int
        or h['aggregate_record_bytes'] != 592 or type(h['aggregate_record_bytes']) is not int
        or h['histogram'] != 'log2_ns_zero_bin'
        or h['detail_policy'] != 'complete_except_aggregated_event_calls'
        or h['aggregate_kinds'] != AGGREGATE_KINDS
        or h['record_bytes'] != 176 or type(h['record_bytes']) is not int):
        raise ValueError('unsupported v2 header/aggregation contract')
    old_footer = {'type','recorded','dropped','unfinished','metadata_truncated','complete'}
    extra_counts = {'aggregate_groups','aggregate_started','aggregate_completed','aggregate_unfinished','aggregate_dropped'}
    if (set(f) != old_footer | extra_counts | {'counter_overflow','invalid_metadata'}
        or any(not integer(f[k]) or f[k] > UINT64_MAX for k in extra_counts)
        or f['counter_overflow'] is not False or f['invalid_metadata'] is not False
        or f['aggregate_unfinished'] or f['aggregate_dropped']
        or f['aggregate_started'] != f['aggregate_completed']):
        raise ValueError('incomplete/invalid aggregate coverage')
    spans, aggregates = [], []
    aggregate_started = False
    for r in records:
        if r.get('type') == 'aggregate':
            aggregate_started = True; aggregates.append(r)
        elif r.get('type') == 'span' and not aggregate_started:
            spans.append(r)
        else:
            raise ValueError('invalid record type/order')
    if len(aggregates) != f['aggregate_groups'] or len(aggregates) > h['aggregate_capacity']:
        raise ValueError('aggregate group count mismatch')
    by_backend = {}; total_count = 0
    for a in aggregates:
        if (set(a) != {'type','kind','backend','count','total_cpu_ns','max_cpu_ns','histogram'}
            or a['kind'] not in AGGREGATE_KINDS or type(a['backend']) is not str
            or not 0 < len(a['backend'].encode()) < 32 or '\x00' in a['backend']
            or any(not integer(a[k]) or a[k] > UINT64_MAX for k in ('count','total_cpu_ns','max_cpu_ns'))
            or a['count'] == 0 or type(a['histogram']) is not list or len(a['histogram']) != 65
            or any(not integer(n) or n > UINT64_MAX for n in a['histogram'])):
            raise ValueError('malformed aggregate')
        key = a['kind'] + ' | ' + a['backend']
        if key in by_backend: raise ValueError('duplicate aggregate key')
        hist = a['histogram']; maximum = a['max_cpu_ns']; total = a['total_cpu_ns']
        low = sum(n * (0 if i == 0 else 2**(i-1)) for i,n in enumerate(hist))
        # At least one call must actually attain the advertised maximum.
        if maximum:
            low += maximum - 2**(maximum.bit_length()-1)
        high = sum(n * min(maximum, 0 if i == 0 else 2**i-1) for i,n in enumerate(hist))
        if (sum(hist) != a['count'] or maximum > total or total > maximum*a['count']
            or not low <= total <= high or hist[maximum.bit_length()] == 0
            or any(hist[i] for i in range(maximum.bit_length()+1,65))):
            raise ValueError('aggregate count/duration/histogram inconsistent')
        by_backend[key] = a; total_count += a['count']
    if total_count != f['aggregate_completed']:
        raise ValueError('aggregate coverage counter mismatch')
    for s in spans:
        if s.get('kind') not in DETAIL_KINDS:
            raise ValueError('unsupported detailed kind or unaggregated event call')
        for key, limit in (('kind',48),('src',32),('dst',32)):
            if type(s.get(key)) is not str or len(s[key].encode()) >= limit or '\x00' in s[key]:
                raise ValueError('invalid detailed metadata')
        if any(not integer(s.get(k)) or s[k] > UINT64_MAX for k in
               ('seq','thread','begin_ns','end_ns','graph','logical_bytes','n_copies')):
            raise ValueError('invalid detailed integer')
        if any(not integer(s.get(k), -1) or s[k] > 2**31-1 for k in ('split','copy','n_copies')):
            raise ValueError('invalid detailed context')
    legacy = [{k: ('heterollm.cpu_trace.v1' if k == 'schema' else h[k]) for k in old_header},
              *spans, {k:f[k] for k in old_footer}]
    result = analyze_v1('\n'.join(map(json.dumps,legacy)), stderr, allow_empty=True)
    result.update(schema='heterollm.cpu_trace.summary.v2', aggregates=by_backend,
                  aggregate_calls=total_count, detailed_events=len(spans),
                  detailed_backend_wait_union_ns=result['cpu_wait_union_ns'], cpu_wait_union_ns=None,
                  all_cpu_wait_union_measurable=False,
                  record_pool_bytes=h['capacity']*h['record_bytes']+h['aggregate_capacity']*h['aggregate_record_bytes'])
    result['limitations'] += [
        'Routine event calls are exact per-backend aggregates, not sampled detailed spans',
        'Aggregate inclusive totals may overlap detailed waits; never add them to infer wall time',
        'Union of all synchronization intervals is UNKNOWN because aggregate timestamps are not retained']
    return result
