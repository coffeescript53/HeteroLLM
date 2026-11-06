"""Argument dialects and diagnostics for pinned llama.cpp 5e03bdd87."""
import json
import re
from .config import digest

PROMPT = 'What is 6 multiplied by 7? Reply with only the integer.'


def bench_command(config, policy, reps, diagnostic=False):
    argv = [config.bench,'-m',config.model,'-dev','/'.join(policy.devices),'-sm','layer',
            '-ngl',str(policy.gpu_layers),'-fitt','0','-p',str(config.prompt_tokens),
            '-n',str(config.generation_tokens),'-d',str(config.depth),'-b',str(config.batch),
            '-ub',str(config.ubatch),'-t',str(config.threads),'-ctk',config.cache_k,'-ctv',config.cache_v,
            '-fa',config.flash_attention,'-nkvo','0','-nopo','0','--no-host','0','--repack','1',
            '-lm','mmap','-lzm','off','-r',str(reps),'-o','json','--offline']
    if policy.tensor_split is not None:
        argv += ['-ts','/'.join(str(x) for x in policy.tensor_split)]
    if diagnostic:
        argv += ['-v']
    return argv


def functional_command(config, policy):
    argv = [config.cli,'-m',config.model,'--device',','.join(policy.devices),'--split-mode','layer',
            '-ngl',str(policy.gpu_layers),'--fit','off','-c','512','-b',str(config.batch),'-ub',str(config.ubatch),
            '-t',str(config.threads),'-ctk',config.cache_k,'-ctv',config.cache_v,'-fa',config.flash_attention,
            '-lm','mmap','-lzm','off','--seed','42','--temp','0','-n','8','--single-turn',
            '--simple-io','--color','off','--no-display-prompt','--no-show-timings','-p',PROMPT,'-v','--offline']
    if policy.tensor_split is not None:
        argv += ['--tensor-split',','.join(str(x) for x in policy.tensor_split)]
    return argv


def require_capabilities(snapshot):
    required = {'bench_help':['--device','--split-mode','--tensor-split','--n-gpu-layers','--fit-target',
                              '--load-mode','--lazy-mode','--n-depth','--cache-type-k','--cache-type-v','--offline'],
                'cli_help':['--device','--split-mode','--tensor-split','--fit','--single-turn','--simple-io','--offline']}
    for key, flags in required.items():
        entry = snapshot[key]
        if entry['exit_code'] != 0:
            raise ValueError(f'{key} failed: {entry["stderr"]}')
        if any(flag not in entry['stdout'] for flag in flags):
            raise ValueError(f'{key}: unsupported runtime controls')
    if '<dev0/dev1/...>' not in snapshot['bench_help']['stdout']:
        raise ValueError('unrecognized llama-bench device-list dialect')


def placement(text):
    assignments = re.findall(r'load_tensors: layer\s+(\d+) assigned to device (\w+)', text)
    layers = {}
    conflicts = []
    for i,d in assignments:
        if i in layers and layers[i] != d:
            conflicts.append(i)
        layers[i] = d
    compute = set()
    # Match only the node's backend (first bracket), never a source tensor's backend.
    for line in text.splitlines():
        m = re.search(r'node #\s*\d+\s+\(\s*([^)]*)\):.*?\[\s*(\w+)\s+[^]]*\].*?c=1:', line)
        if m and m[1].strip().startswith(('MUL_MAT','MUL_MAT_ID')):
            compute.add(m[2])
    buffers = {}
    for d, mb in re.findall(r'\b(CUDA\d+|ROCm\d+|CPU(?:_Mapped)?)\s+model buffer size\s*=\s*([\d.]+) MiB',text):
        buffers[d] = max(buffers.get(d,0),float(mb))
    contexts = [int(n) for n in re.findall(r'\bn_ctx\s*=\s*(\d+)',text)]
    kv_lines = [line for line in text.splitlines() if 'KV buffer size' in line or 'KV self size' in line]
    return {'layers':layers,'conflicting_layers':sorted(set(conflicts)), 'matrix_compute_backends':sorted(compute),
            'model_buffers_mib':buffers,'contexts':contexts,'kv_lines':kv_lines,
            'interpretation':'scheduler assignment plus successful decode, not a kernel timing trace'}


def verify_placement(evidence, policy, layout):
    errors = []
    layers = evidence['layers']
    if set(layers) != {str(i) for i in range(layout.blocks+1)} or evidence['conflicting_layers']:
        errors.append('complete consistent layer mapping was not observed')
    if set(layers.values()) - (set(policy.devices) | {'CPU'}):
        errors.append('unexpected device in layer mapping')
    for device in policy.devices:
        if device not in layers.values() or device not in evidence['matrix_compute_backends']:
            errors.append(f'{device}: assigned layers and matrix-operation scheduling required')
        if evidence['model_buffers_mib'].get(device,0) <= 0:
            errors.append(f'{device}: positive model buffer not observed')
    if not evidence['contexts'] or not evidence['kv_lines']:
        errors.append('effective context/KV diagnostics missing')
    return errors


def response_correct(text, stderr=''):
    """Validate an entire answer, or the characterized pinned CLI single-turn UI.

    Framed stdout requires one final (not partial) structured assistant message
    from common/chat.cpp and agreement with the entire displayed response.
    Never search for an isolated "42" or strip arbitrary explanatory text.
    """
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate message field')
            result[key] = value
        return result

    records = [line for line in stderr.split('\n') if 'Parsed message:' in line]
    message = None
    if records:
        if len(records) != 1:
            return False
        match = re.fullmatch(r'\d+\.\d{2}\.\d{3}\.\d{3} D Parsed message: (.+)', records[0])
        if not match:
            return False
        try:
            message = json.loads(match[1], object_pairs_hook=unique_object)
        except (ValueError, TypeError):
            return False
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] != 'assistant' or not isinstance(message['content'], str)):
            return False

    answer = text
    if text.startswith('\n\nLoading model... \n\n'):
        prompt = '\n> ' + PROMPT + '\n'
        footer = '\n\n\nExiting...\n'
        menu = ('\n\navailable commands:\n'
                '  /exit or Ctrl+C     stop or exit\n'
                '  /regen              regenerate the last response\n'
                '  /clear              clear the chat history\n'
                '  /read <file>        add a text file\n'
                '  /glob <pattern>     add text files using globbing pattern\n\n\n')
        if message is None or text.count(prompt) != 1 or not text.endswith(footer):
            return False
        header, answer = text.split(prompt)
        if (not header.endswith(menu) or '\x1b' in header
                or not re.search(r'\nbuild      : b11434-5e03bdd87\nmodel      : [^\r\n]+'
                                 r'\nftype      : [^\r\n]+\nmodalities : text\n', header)):
            return False
        answer = answer[:-len(footer)]

    answer = answer.strip(' \t\r\n')
    return answer == '42' and (message is None or message['content'].strip(' \t\r\n') == answer)


def validation_key(config, policy, model_identity, runtime_files):
    return digest({'policy':policy.to_dict(),'workload':config.workload(),'model':model_identity,
                   'runtime':runtime_files,'environment':config.environment, 'devices':config.devices})
