"""Public CLI. Preview is the default for every inference operation."""
import argparse
import json
from pathlib import Path
import shlex
import sys

from . import __version__, analysis
from .config import Config, Policy
from .gguf import planning_placeholder, read_layout
from .hardware import discover
from .planner import candidates
from .runner import (Session, StopSuite, execution_lock, load_receipt, unique_directory,
                     verify_model, write_json)
from .runtime import bench_command, functional_command, validation_key


def emit(value):
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


def prepare(config, snapshot_path=None):
    snapshot = json.loads(Path(snapshot_path).read_text()) if snapshot_path else discover(config)
    layout = read_layout(config.model) if Path(config.model).is_file() else planning_placeholder(config.model_size_bytes)
    return snapshot,layout,candidates(layout,snapshot,config)


def get_policy(value, choices):
    if value.endswith('.json'):
        data = json.loads(Path(value).read_text())
        return Policy.from_dict(data.get('policy',data.get('selected_policy',data)))
    for c in choices:
        if c.policy.name==value:
            return c.policy
    raise ValueError(f'unknown policy {value!r}')


def preview(config, choices, name=None):
    chosen = choices if name is None else [c for c in choices if c.policy.name==name]
    if not chosen and name:
        policy = get_policy(name,choices)
        chosen = [type('PreviewCandidate',(),{'policy':policy,'admissible':False,'reasons':['external policy requires live screening']})()]
    return {'dry_run':True,'model_present':Path(config.model).exists(),'maximum_candidates':6,
            'maximum_launches':config.max_launches,'suite_deadline_seconds':config.suite_seconds,
            'environment_overrides':config.environment,'output_root':config.output,'workload':config.workload(),
            'candidates':[{'policy':c.policy.to_dict(),'admissible':c.admissible,'reasons':c.reasons,
                           'functional_argv':functional_command(config,c.policy),
                           'diagnostic_argv':bench_command(config,c.policy,1,True),
                           'exploratory_argv':bench_command(config,c.policy,config.exploratory_reps),
                           'final_argv':bench_command(config,c.policy,config.final_reps)} for c in chosen]}


def export_result(selection_path, output):
    source = Path(selection_path)
    result = json.loads(source.read_text())
    selected = result.get('selected')
    if not selected or 'selected_policy' not in result:
        raise ValueError('no selected measured policy to export')
    checked = analysis.selection(result['records'])
    if selected != checked['selected']:
        raise ValueError('selection does not match saved independent-launch evidence')
    policies = [r['policy'] for r in result['records'] if r['policy']['name']==selected and r['status']=='ok']
    if not policies or any(p != result['selected_policy'] for p in policies):
        raise ValueError('export policy differs from measured policy')
    Policy.from_dict(result['selected_policy'])
    folder = unique_directory(output,'export')
    write_json(folder/'policy.json',{'policy':result['selected_policy'],'selection_source':str(source.resolve()),
                                    'workload_and_runtime_config':json.loads((source.parent/'config.json').read_text()),
                                    'claim':checked['claim']})
    lines = ['# Observed placement search result','',f'Selected: **{selected}**. Objective: independent-launch mean TG tokens/s.','',
             '| Policy | Launches | PP mean tokens/s | TG mean tokens/s | TG between-launch SD |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name,g in checked['groups'].items():
        lines.append(f'| {name} | {g["tg"]["n"]} | {g["pp"]["mean"]:.3f} | {g["tg"]["mean"]:.3f} | {g["tg"]["sample_sd"]} |')
    lines += ['',f'Strongest remeasured baseline: {checked["strongest_baseline"]}.',
              '', 'Comparisons:','', '```json',json.dumps(checked['comparisons'],indent=2),'```',
              '', 'Diagnostic placement is captured separately; quiet runs do not directly trace their final mapping.',
              'Small final sample size, selection bias, and residual order/temperature effects limit inference.',
              f'Raw evidence index: {source.resolve()}','']
    (folder/'report.md').write_text('\n'.join(lines))
    return str(folder)


def parser():
    p = argparse.ArgumentParser(prog='heterollm',description='Constrained empirical layer-placement optimizer; upstream llama.cpp executes inference.')
    p.add_argument('--version',action='version',version=__version__)
    sub = p.add_subparsers(dest='command',required=True)
    for name in ('inspect','policies','preview','validate','benchmark','tune','verify-model'):
        s = sub.add_parser(name)
        s.add_argument('--config',default='configs/llama33-70b.json')
        if name != 'verify-model':
            s.add_argument('--snapshot',help='read a saved snapshot; execution is forbidden with this option')
        if name in ('preview','validate','benchmark'):
            s.add_argument('--policy',default='mixed-default',help='candidate name or exported policy JSON')
        if name in ('validate','benchmark','tune'):
            s.add_argument('--receipt',help='full model SHA-256 verification receipt')
            action = s.add_mutually_exclusive_group()
            action.add_argument('--execute',action='store_true',help='run only after owner approval of the bounded plan')
            action.add_argument('--dry-run',action='store_true',help='print plan without starting inference (default)')
        if name=='benchmark':
            s.add_argument('--validation',help='matching validation.json from a successful functional and placement test')
        if name=='verify-model':
            s.add_argument('--receipt',required=True,help='new, nonexisting receipt file')
    s = sub.add_parser('compare')
    s.add_argument('experiments',nargs='+',help='suite directories or selection.json files')
    s = sub.add_parser('export')
    s.add_argument('selection')
    s.add_argument('--output',default='exports')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command=='compare':
            records = []
            for value in args.experiments:
                p = Path(value)
                if p.is_file():
                    records.extend(json.loads(p.read_text())['records'])
                else:
                    records.extend(json.loads(f.read_text()) for f in sorted(p.glob('*/record.json')))
            emit(analysis.selection(records)); return 0
        if args.command=='export':
            print(export_result(args.selection,args.output)); return 0
        config = Config.load(args.config)
        if args.command=='verify-model':
            emit(verify_model(config,args.receipt)); return 0
        if args.command=='inspect':
            snapshot = json.loads(Path(args.snapshot).read_text()) if args.snapshot else discover(config)
            emit(snapshot); return 0
        snapshot,layout,choices = prepare(config,args.snapshot)
        if args.command=='policies':
            emit({'metadata_exact':layout.exact,'candidates':[c.to_dict() for c in choices]}); return 0
        if args.command=='preview' or not args.execute:
            emit(preview(config,choices,getattr(args,'policy',None))); return 0
        if args.snapshot:
            raise ValueError('saved snapshots cannot authorize execution; use live discovery')
        if not args.receipt:
            raise ValueError('--receipt is required for execution; run verify-model once after approved download')
        with execution_lock():
            session = Session(config,args.receipt)
            print(f'Evidence directory: {session.directory}',file=sys.stderr,flush=True)
            if args.command=='tune':
                result = session.tune()
            else:
                policy = get_policy(args.policy,choices)
                if args.command=='validate':
                    result = session.validate(policy)
                else:
                    if not args.validation:
                        raise ValueError('--validation is required')
                    valid = json.loads(Path(args.validation).read_text())
                    if valid['status']!='ok' or valid['validation_key']!=validation_key(config,policy,session.model,session.runtime_files):
                        raise ValueError('validation does not match this model/runtime/policy/workload')
                    session.validations[policy.name] = valid
                    result = session.benchmark(policy,'final',config.final_reps)
            emit(result)
            return 0 if result.get('status','ok')=='ok' and result.get('selected',True) is not None else 1
    except (ValueError,OSError,TypeError,KeyError,StopSuite) as e:
        print(f'heterollm: {e}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
