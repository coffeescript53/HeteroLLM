"""Strict, portable configuration. Relative paths resolve from the config file."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math
import re

GIB = 1024**3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')


@dataclass(frozen=True)
class Policy:
    name: str
    devices: tuple[str, ...]
    tensor_split: tuple[float, ...] | None
    gpu_layers: int
    split_mode: str = 'layer'
    fit_target_mib: int = 0
    host_offload: str = 'remaining_layers_and_embeddings'

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9-]{0,47}', self.name):
            raise ValueError('invalid policy name')
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError('devices must be nonempty and unique')
        if any(not re.fullmatch(r'(CUDA|ROCm)\d+', d) for d in self.devices):
            raise ValueError('only explicit CUDA/ROCm devices are supported')
        if self.split_mode != 'layer' or self.fit_target_mib != 0:
            raise ValueError('this protocol requires layer splitting with fitting disabled')
        if self.host_offload != 'remaining_layers_and_embeddings':
            raise ValueError('unsupported host policy')
        integer(self.gpu_layers, 'gpu_layers', 0, 999)
        if self.tensor_split is not None:
            if len(self.tensor_split) != len(self.devices):
                raise ValueError('one tensor proportion is required per device')
            if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in self.tensor_split):
                raise ValueError('tensor proportions must be positive finite numbers')

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['devices'] = tuple(value['devices'])
        if value.get('tensor_split') is not None:
            value['tensor_split'] = tuple(value['tensor_split'])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Config:
    model: str
    bench: str
    cli: str
    source: str
    output: str
    model_sha256: str
    model_size_bytes: int
    source_revision: str
    devices: dict[str, str]
    environment: dict[str, str] = field(default_factory=dict)
    prompt_tokens: int = 512
    generation_tokens: int = 128
    depth: int = 0
    batch: int = 512
    ubatch: int = 128
    threads: int = 16
    cache_k: str = 'f16'
    cache_v: str = 'f16'
    flash_attention: str = 'off'
    gpu_reserve_mib: int = 3072
    host_reserve_mib: int = 8192
    timeout_seconds: int = 900
    suite_seconds: int = 10800
    max_launches: int = 30
    max_temperature_c: int = 85
    max_swap_growth_mib: int = 256
    minimum_available_ram_mib: int = 4096
    max_log_mib: int = 256
    exploratory_reps: int = 2
    final_reps: int = 5
    final_rounds: int = 3

    def __post_init__(self):
        for key in ('model_sha256', 'source_revision'):
            if not re.fullmatch(r'[0-9a-f]{64}' if key == 'model_sha256' else r'[0-9a-f]{40}', getattr(self, key)):
                raise ValueError(f'{key} must be a full hexadecimal digest')
        bounds = {'model_size_bytes': (1, 1024**4), 'prompt_tokens': (1, 4096), 'generation_tokens': (1, 1024),
                  'depth': (0, 4096), 'batch': (1, 4096), 'ubatch': (1, 4096), 'threads': (1, 256),
                  'gpu_reserve_mib': (1024, 32768), 'host_reserve_mib': (4096, 262144),
                  'timeout_seconds': (1, 1800), 'suite_seconds': (1, 14400), 'max_launches': (1, 30),
                  'max_temperature_c': (40, 90), 'max_swap_growth_mib': (0, 256),
                  'minimum_available_ram_mib': (1024, 32768), 'max_log_mib': (1, 512),
                  'exploratory_reps': (1, 5), 'final_reps': (2, 10), 'final_rounds': (3, 5)}
        for name, (low, high) in bounds.items():
            integer(getattr(self, name), name, low, high)
        if self.ubatch > self.batch:
            raise ValueError('ubatch cannot exceed batch')
        if self.cache_k != 'f16' or self.cache_v != 'f16' or self.flash_attention != 'off':
            raise ValueError('initial controlled protocol supports F16 KV and FA off only')
        if set(self.devices) != {'CUDA0', 'ROCm0'}:
            raise ValueError('initial search requires explicitly mapped CUDA0 and ROCm0')
        if len(set(self.devices.values())) != 2 or any(not re.fullmatch(r'0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', p) for p in self.devices.values()):
            raise ValueError('provide two unique canonical PCI addresses')
        if set(self.environment) - {'LD_LIBRARY_PATH', 'ROCM_PATH'}:
            raise ValueError('only library-location environment overrides are supported')
        if any(not isinstance(v, str) or '\x00' in v for v in self.environment.values()):
            raise ValueError('environment paths must be strings without NUL')
        if any(not isinstance(getattr(self, p), str) or not getattr(self, p) for p in ('model','bench','cli','source','output')):
            raise ValueError('paths must be nonempty strings')

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        for key in ('model', 'bench', 'cli', 'source', 'output'):
            data[key] = str((path.parent / data[key]).resolve())
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def workload(self):
        names = ('prompt_tokens','generation_tokens','depth','batch','ubatch','threads','cache_k','cache_v','flash_attention')
        return {**{k: getattr(self, k) for k in names}, 'load_mode': 'mmap', 'lazy_mode': 'off', 'warmup': 'upstream_default', 'fit_target_mib': 0}
