"""Read only the GGUF directory. Never mmap or read weight payloads here."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import struct

# GGML block elements and encoded bytes for the supported dense-Llama quants.
TYPE_SIZES = {0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24),
              8:(32,34), 9:(32,40), 10:(256,84), 11:(256,110), 12:(256,144),
              13:(256,176), 14:(256,210), 15:(256,292), 30:(1,2)}
SCALARS = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}


@dataclass(frozen=True)
class ModelLayout:
    size_bytes: int
    layer_bytes: tuple[int, ...]  # repeating blocks followed by output group
    host_bytes: int
    kv_bytes_per_layer_token: int
    architecture: str = 'llama'
    exact: bool = True

    @property
    def blocks(self):
        return len(self.layer_bytes) - 1

    def to_dict(self):
        return asdict(self)


class DirectoryReader:
    def __init__(self, stream):
        self.f = stream

    def read(self, count):
        if count < 0 or self.f.tell() + count > 128 * 1024**2:
            raise ValueError('GGUF header exceeds the 128 MiB inspection limit')
        b = self.f.read(count)
        if len(b) != count:
            raise ValueError('truncated GGUF header')
        return b

    def scalar(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep=True):
        n = self.scalar('Q')
        data = self.read(n)
        return data.decode('utf-8') if keep else None

    def value(self, kind, keep=True, depth=0):
        if depth > 2:
            raise ValueError('excessive GGUF nesting')
        if kind in SCALARS:
            v = self.scalar(SCALARS[kind])
            return v if keep else None
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            subtype, n = self.scalar('I'), self.scalar('Q')
            if n > 2_000_000:
                raise ValueError('excessive GGUF array')
            for _ in range(n):
                self.value(subtype, False, depth + 1)
            return None
        raise ValueError(f'unsupported GGUF metadata type {kind}')


def read_layout(path):
    path = Path(path)
    with path.open('rb') as stream:
        r = DirectoryReader(stream)
        if r.read(4) != b'GGUF' or r.scalar('I') != 3:
            raise ValueError('expected little-endian GGUF v3')
        nt, nk = r.scalar('Q'), r.scalar('Q')
        if nt > 100_000 or nk > 100_000:
            raise ValueError('unreasonable GGUF directory counts')
        meta = {}
        for _ in range(nk):
            key, kind = r.string(), r.scalar('I')
            keep = key.startswith('llama.') or key in ('general.architecture','general.alignment','split.count')
            val = r.value(kind, keep)
            if keep:
                meta[key] = val
        if meta.get('general.architecture') != 'llama' or meta.get('split.count', 1) != 1:
            raise ValueError('this optimizer currently supports single-file dense Llama GGUF only')
        blocks = meta['llama.block_count']
        if type(blocks) is not int or not 1 <= blocks <= 512:
            raise ValueError('invalid Llama block count')
        groups, host, tensors = [0] * (blocks + 1), 0, []
        for _ in range(nt):
            name, nd = r.string(), r.scalar('I')
            if not 1 <= nd <= 4:
                raise ValueError('invalid tensor rank')
            shape = [r.scalar('Q') for _ in range(nd)]
            qt, offset = r.scalar('I'), r.scalar('Q')
            if qt not in TYPE_SIZES:
                raise ValueError(f'unsupported tensor type {qt}; cannot safely estimate bytes')
            elements, encoded = TYPE_SIZES[qt]
            if any(x <= 0 for x in shape) or shape[0] % elements:
                raise ValueError('invalid quantized tensor dimensions')
            size = math.prod(shape) // elements * encoded
            tensors.append((offset, size))
            match = re.match(r'blk\.(\d+)\.', name)
            if match:
                idx = int(match[1])
                if idx >= blocks:
                    raise ValueError('tensor layer outside block count')
                groups[idx] += size
            elif name.startswith(('output.', 'output_norm.')):
                groups[-1] += size
            else:
                host += size
        alignment = meta.get('general.alignment', 32)
        if type(alignment) is not int or alignment < 1 or alignment > 4096:
            raise ValueError('invalid alignment')
        data_start = (r.f.tell() + alignment - 1) // alignment * alignment
        file_size = path.stat().st_size
        previous_end = 0
        for offset, size in sorted(tensors):
            if offset < previous_end or data_start + offset + size > file_size:
                raise ValueError('overlapping or truncated GGUF tensor payload')
            previous_end = offset + size
        if not all(groups):
            raise ValueError('missing layer/output tensors; tied output is not supported by this estimator')
        heads, kv_heads = meta['llama.attention.head_count'], meta['llama.attention.head_count_kv']
        width = meta['llama.embedding_length']
        if any(type(n) is not int or n <= 0 for n in (heads,kv_heads,width)) or width % heads:
            raise ValueError('invalid attention geometry')
        if meta.get('llama.expert_count',0):
            raise ValueError('MoE Llama is outside this dense-model estimator')
        key_dim = meta.get('llama.attention.key_length', width // heads)
        value_dim = meta.get('llama.attention.value_length', width // heads)
        if any(type(n) is not int or n <= 0 for n in (key_dim,value_dim)):
            raise ValueError('invalid KV dimensions')
        kv = kv_heads * (key_dim + value_dim) * 2  # F16 K and V
        return ModelLayout(file_size, tuple(groups), host, kv)


def planning_placeholder(size):
    """Pre-download illustration only; never admitted for execution."""
    group = size // 82
    return ModelLayout(size, (group,) * 81, size - group * 81, 4096, exact=False)
