"""Strict phase/device-event analysis; never treats device clocks as common."""
from collections import Counter, defaultdict
import math
from .trace_analysis import analyze, strict_json, integer

HEADER={'type':'header','schema':'heterollm.timing.v3','clock':'steady_clock_ns',
        'capacity':16384,'pool_pairs_per_backend':256,
        'correlation':'cpu_enqueue_to_completion_enclosure','extra_synchronization':False}
PHASES=['warmup_pp','pp512','warmup_tg','tg128']
DEVICE_KEYS={'type','id','phase','graph','split','thread','logical_bytes','start_before',
             'start_after','end_before','end_after','observed','elapsed_ms','kind','backend','stream'}
KINDS={'graph_uncaptured','graph_replay','copy_h2d','copy_d2h','copy_h2d_2d','copy_d2h_2d'}

def analyze_timing(timing_text,cpu_text,stderr,*,repetitions=1):
    if type(repetitions) is not int or repetitions not in (1,3):
        raise ValueError('unsupported timing repetitions')
    expected=[('warmup_pp',0),*[('pp512',i) for i in range(repetitions)],
              ('warmup_tg',0),*[('tg128',i) for i in range(repetitions)]]
    cpu=analyze(cpu_text,stderr)
    if 'HETEROLLM_TIMING_ERROR' in stderr:raise ValueError('timing recorder error')
    if len(timing_text.encode())>32*1024**2:raise ValueError('timing evidence cap')
    rows=[strict_json(line) for line in timing_text.splitlines()]
    if len(rows)<10 or any(type(r)is not dict for r in rows) or rows[0]!=HEADER or rows[0]['extra_synchronization'] is not False:
        raise ValueError('timing header/records')
    foot=rows[-1]
    if set(foot)!={'type','phases','started','completed','pending','active_phase','invalid','complete'} or foot.get('type')!='footer':raise ValueError('timing footer')
    if foot['complete'] is not True or foot['invalid'] is not False or any(not integer(foot[k]) for k in ('phases','started','completed','pending','active_phase')) or foot['pending'] or foot['active_phase'] or foot['started']!=foot['completed'] or foot['started']>16384:
        raise ValueError('incomplete device capture')
    phases={};devices=[];active=None;last_end=0
    for row in rows[1:-1]:
        if row.get('type')=='phase_begin':
            if set(row)!={'type','phase','cpu_ns','repetition','name'} or active is not None or not integer(row['phase'],1) or row['phase']!=len(phases)+1 or not integer(row['cpu_ns'],1) or row['cpu_ns']<last_end or type(row['repetition'])is not int or len(phases)>=len(expected) or (row['name'],row['repetition'])!=expected[len(phases)]:raise ValueError('phase begin')
            active=row['phase'];phases[active]={'name':row['name'],'begin':row['cpu_ns'],'end':None}
            if repetitions!=1:phases[active]['repetition']=row['repetition']
        elif row.get('type')=='phase_end':
            if set(row)!={'type','phase','cpu_ns'} or not integer(row['phase'],1) or row['phase']!=active or not integer(row['cpu_ns'],1) or row['cpu_ns']<phases[active]['begin']:raise ValueError('phase end')
            phases[active]['end']=row['cpu_ns'];last_end=row['cpu_ns'];active=None
        elif row.get('type')=='device':
            if set(row)!=DEVICE_KEYS or not integer(row['phase'],1) or row['phase']!=active:raise ValueError('device phase/schema')
            for k in ('id','graph','thread','logical_bytes','start_before','start_after','end_before','end_after','observed'):
                if not integer(row[k]) or row[k]>2**64-1:raise ValueError('device integer '+k)
            if not integer(row['split'],-1) or row['kind'] not in KINDS or row['backend'] not in ('CUDA0','ROCm0') or row['stream'] not in ('main','per_thread'):raise ValueError('device metadata')
            times=[row[k] for k in ('start_before','start_after','end_before','end_after','observed')]
            if times!=sorted(times) or times[0]<phases[active]['begin']:raise ValueError('CPU enclosure order')
            ms=row['elapsed_ms']
            if type(ms)not in (int,float) or not math.isfinite(ms) or ms<0 or ms*1e6>times[-1]-times[0]+1e6:raise ValueError('device duration outside enclosure')
            if row['kind'].startswith('graph') and (row['logical_bytes'] or row['stream']!='main' or row['graph']==0 or row['split']<0):raise ValueError('graph metadata')
            if row['kind'].startswith('copy') and (row['logical_bytes']==0 or row['stream']!='per_thread'):raise ValueError('copy metadata')
            devices.append(row)
        else:raise ValueError('unexpected timing record')
    if active is not None or [p['name'] for p in phases.values()]!=[p[0] for p in expected] or foot['phases']!=len(expected) or foot['completed']!=len(devices) or sorted(r['id'] for r in devices)!=list(range(len(devices))):raise ValueError('incomplete phases/device IDs')
    for r in devices:
        if r['observed']>phases[r['phase']]['end']:raise ValueError('completion escaped phase')
    spans=[strict_json(line) for line in cpu_text.splitlines() if line]
    spans=[r for r in spans if r['type']=='span']
    def phase_of(r):
        found=[i for i,p in phases.items() if p['begin']<=r['begin_ns']<=r['end_ns']<=p['end']]
        return found[0] if len(found)==1 else None
    graph_cpu=Counter();graph_gpu=Counter()
    grouped=defaultdict(lambda:{'detailed_cpu_kinds':{},'device_operations':{}})
    for r in spans:
        phase=phase_of(r)
        if phase:
            k=grouped[phases[phase]['name']]['detailed_cpu_kinds'].setdefault(r['kind'],{'count':0,'inclusive_cpu_ns':0})
            k['count']+=1;k['inclusive_cpu_ns']+=r['end_ns']-r['begin_ns']
        if r['kind']=='graph.submit' and r['src'] in ('CUDA0','ROCm0'):
            if phase is None:raise ValueError('unphased GPU graph submission')
            graph_cpu[(phase,r['thread'],r['graph'],r['split'],r['src'])]+=1
    for r in devices:
        if r['kind'].startswith('graph'):graph_gpu[(r['phase'],r['thread'],r['graph'],r['split'],r['backend'])]+=1
        key=r['backend']+' | '+r['kind']
        d=grouped[phases[r['phase']]['name']]['device_operations'].setdefault(key,{'count':0,'completed_stream_elapsed_ms_sum':0,'logical_bytes':0,'max_cpu_alignment_enclosure_ns':0})
        d['count']+=1;d['completed_stream_elapsed_ms_sum']+=r['elapsed_ms'];d['logical_bytes']+=r['logical_bytes'];d['max_cpu_alignment_enclosure_ns']=max(d['max_cpu_alignment_enclosure_ns'],r['observed']-r['start_before'])
    if graph_cpu!=graph_gpu or not graph_gpu:raise ValueError('missing/extra GPU graph timing')
    for phase in phases:
        if not all(any(k[0]==phase and k[-1]==d for k in graph_gpu) for d in ('CUDA0','ROCm0')):raise ValueError('phase lacks both GPU backends')
    matched=set()
    for s in spans:
        if s['kind'] not in ('copy.host_stage.get','copy.host_stage.set'):continue
        backend=s['src'] if s['kind'].endswith('get') else s['dst'];kind='copy_d2h' if s['kind'].endswith('get') else 'copy_h2d'
        if backend not in ('CUDA0','ROCm0'):raise ValueError('unsupported host-stage backend')
        candidates=[r for r in devices if r['id'] not in matched and r['kind']==kind and r['backend']==backend and r['phase']==phase_of(s) and r['thread']==s['thread'] and r['graph']==s['graph'] and r['split']==s['split'] and r['logical_bytes']==s['logical_bytes'] and s['begin_ns']<=r['start_before']<=r['end_after']<=r['observed']<=s['end_ns']]
        if len(candidates)!=1:raise ValueError('missing/ambiguous completed host-stage leg')
        matched.add(candidates[0]['id'])
    return {'schema':'heterollm.timing.summary.v3','complete':True,'phases':phases,'per_phase':dict(grouped),'device_records':len(devices),'matched_host_stage_legs':len(matched),'cpu_trace_complete':cpu['complete'],
            'correlation':'Each device interval is enclosed in CPU [start_before, observed]; width is conservative alignment uncertainty, NOT a calibrated affine clock mapping.',
            'gpu_kernel_only_time':None,'cross_device_overlap':None,'queue_delay':None,'dominant_bottleneck':None,'aggregate_event_phase':None,
            'limitations':['Same-device event elapsed intervals include dependencies and gaps; not kernel-only activity.','Nested CPU and device sums are not independent wall costs.','Device durations are never subtracted across CUDA/HIP clock domains.','No physical PCIe traffic measurement.','CPU event aggregates remain whole-process; detailed routing is phase-attributed.','Loading and initialization are outside device-event coverage.','A 1 ms consistency margin is validation slack, not calibrated event accuracy.']}
