"""Conservative capacity screening, not a latency model or allocation guarantee."""
from bisect import bisect_right
from dataclasses import asdict, dataclass
from itertools import accumulate
from .config import Policy
from .hardware import MIB


@dataclass
class Candidate:
    policy: Policy
    admissible: bool
    reasons: list[str]
    gpu_estimated_bytes: dict[str, int]
    host_estimated_bytes: int
    estimated_layers: dict[str, list[int]]
    metadata_exact: bool
    priority: int

    def to_dict(self):
        return asdict(self)


def estimate(policy, layout, snapshot, config):
    devices = {d['name']:d for d in snapshot['devices']}
    missing = set(policy.devices) - devices.keys()
    if missing:
        return Candidate(policy,False,[f'runtime devices unavailable: {sorted(missing)}'],{},0,{},layout.exact,99)
    ratios = policy.tensor_split or tuple(devices[d]['free_bytes'] for d in policy.devices)
    if sum(ratios) <= 0:
        return Candidate(policy,False,['no reported free device memory'],{},0,{},layout.exact,99)
    boundaries = list(accumulate(x / sum(ratios) for x in ratios))
    count = min(policy.gpu_layers, len(layout.layer_bytes))
    start = len(layout.layer_bytes) - count
    assigned = {d:[] for d in policy.devices}
    weights = {d:0 for d in policy.devices}
    host = layout.host_bytes
    # llama.cpp uses upper_bound((layer - gpu_start) / active_gpu_layers).
    for layer, size in enumerate(layout.layer_bytes):
        if layer < start:
            host += size
        else:
            idx = min(bisect_right(boundaries, (layer-start)/count),len(policy.devices)-1)
            d = policy.devices[idx]
            assigned[d].append(layer)
            weights[d] += size
    context = max(config.prompt_tokens,config.generation_tokens,512) + config.depth
    context = (context + 255)//256*256  # conservative allocation padding
    kv_per_layer = layout.kv_bytes_per_layer_token * context
    for d in weights:
        weights[d] += sum(layer < layout.blocks for layer in assigned[d]) * kv_per_layer
    host += min(start,layout.blocks) * kv_per_layer
    reasons = []
    for d in policy.devices:
        if devices[d].get('pci') != config.devices[d]:
            reasons.append(f'{d}: physical PCI identity missing or mismatched')
        budget = devices[d]['free_bytes'] - config.gpu_reserve_mib*MIB
        if weights[d] > budget:
            reasons.append(f'{d}: estimated weights/KV exceed free VRAM minus reserve by {weights[d]-budget} bytes')
    available = snapshot['telemetry']['memory']['MemAvailable']
    if host > available-config.host_reserve_mib*MIB:
        reasons.append('estimated CPU weights/KV exceed available RAM minus host reserve')
    if not layout.exact:
        reasons.append('pre-download illustration: actual GGUF directory is required before execution')
    return Candidate(policy, not reasons, reasons, weights, host, assigned,layout.exact,0)


def candidates(layout, snapshot, config):
    result = []
    for name, device in [('cuda-only','CUDA0'),('rocm-only','ROCm0')]:
        chosen = None
        for n in range(len(layout.layer_bytes),-1,-1):
            c = estimate(Policy(name,(device,),None,n), layout,snapshot,config)
            # Pre-download candidates may be illustrated but never executed.
            if not [r for r in c.reasons if not r.startswith('pre-download')]:
                chosen = c
                break
        result.append(chosen or c)
    for name, ratio in [('mixed-default',None),('mixed-equal',(1,1)),('mixed-3-2',(3,2)),('mixed-2-1',(2,1))]:
        result.append(estimate(Policy(name,('CUDA0','ROCm0'),ratio,999),layout,snapshot,config))
    priority = {'mixed-default':0,'cuda-only':1,'rocm-only':2,'mixed-equal':3,'mixed-3-2':4,'mixed-2-1':5}
    for c in result:
        c.priority = priority[c.policy.name]
    return sorted(result,key=lambda c:(c.priority,c.policy.name))


def balanced_order(names, rounds):
    """Rotated/reversed rounds reduce order bias; three rounds are not full balance."""
    names = sorted(set(names))
    if not names:
        return []
    result = []
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        result.append(order if r % 2 == 0 else list(reversed(order)))
    return result
