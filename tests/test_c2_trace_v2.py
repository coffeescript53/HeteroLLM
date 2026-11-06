"""Bounded CPU-only recorder pressure, concurrency, exact aggregates and loss tests."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from heterollm.trace_analysis import analyze
from helpers import ROOT

HARNESS = r'''
#include "heterollm-cpu-trace.h"
#include <thread>
#include <vector>
#include <string>
#include <limits>
int main(int argc, char **argv) {
    std::string mode=argc>1?argv[1]:"mixed";
    int changed=0;
    if(mode=="disabled") {
        HT_TRACE_EVENT((++changed,"event.synchronize"),(++changed,"CUDA0"));
        HT_TRACE_SCOPE("copy.host_stage",(++changed,"CUDA0"),"ROCm0",++changed);
        return changed;
    }
#if defined(HETEROLLM_CPU_TRACE) && HETEROLLM_CPU_TRACE
    auto & r=heterollm_trace::recorder();
    if(mode=="unfinished") { new heterollm_trace::AggregateSpan("event.synchronize","CUDA0"); }
    else if(mode=="detail_unfinished") { new heterollm_trace::Span("graph.submit","CUDA0",""); }
    else if(mode=="aggregate_overflow") {
        for(int i=0;i<33;i++) { auto s=r.aggregate_begin("event.synchronize",("GPU"+std::to_string(i)).c_str());r.aggregate_end(s,1); }
    } else if(mode=="counter_overflow") {
        for(int i=0;i<2;i++){auto s=r.aggregate_begin("event.synchronize","CUDA0");r.aggregate_end(s,std::numeric_limits<uint64_t>::max());}
    } else if(mode=="metadata") { HT_TRACE_EVENT("event.synchronize","abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"); }
    else if(mode=="bad_kind") { HT_TRACE_EVENT("not.allowed","CUDA0"); }
    else if(mode=="critical_overflow") {
        for(int i=0;i<33;i++){ HT_TRACE_SCOPE("graph.submit","CUDA0","",0); }
    } else if(mode=="abrupt") { HT_TRACE_EVENT("event.synchronize","CUDA0"); _exit(7); }
    else if(mode=="aggregate_only") { HT_TRACE_EVENT("event.synchronize","CUDA0"); }
    else if(mode=="write_failure") {
        { HT_TRACE_SCOPE("graph.submit","CUDA0","",0); }
        // Recorder owns the first newly opened fd in this isolated fixture.
        // Force the final fflush/fsync/close failure without touching host files.
        for(int fd=3;fd<32;fd++)close(fd);
    } else {
        std::vector<std::thread> threads;
        for(int t=0;t<4;t++)threads.emplace_back([&,t]{
            const char * backend=t%2?"ROCm0":"CUDA0";
            for(int i=0;i<25000;i++) {
                auto s=r.aggregate_begin(i%2?"event.synchronize":"event.record_submit",backend);
                r.aggregate_end(s,uint64_t(i%8)); // Exact known durations, no wall-clock assertion.
                if(i%1000==0) {
                    HT_TRACE_GRAPH(1,0); HT_TRACE_SPLIT(1);
                    HT_TRACE_SCOPE("graph.compute_splits","","",0);
                    { HT_TRACE_SCOPE("graph.allocate","","",0); }
                    { HT_TRACE_SCOPE("graph.submit",backend,"",0); }
                    { HT_TRACE_SCOPE("copy.scheduler_async_attempt","CUDA0","ROCm0",16); }
                    { HT_TRACE_SCOPE("copy.scheduler_fallback","CUDA0","ROCm0",16);
                      { HT_TRACE_SCOPE("backend.synchronize","CUDA0","",0); }
                      HT_TRACE_SCOPE("copy.host_stage","CUDA0","ROCm0",16);
                      { HT_TRACE_SCOPE("copy.host_stage.get","CUDA0","host_malloc",16); }
                      { HT_TRACE_SCOPE("copy.host_stage.set","host_malloc","ROCm0",16); }
                    }
                }
            }
        });
        for(auto & t:threads)t.join();
    }
#endif
}
'''


def fixture_v2():
    from test_c2_trace_analysis import fixture
    rows=fixture(); rows[0].update(schema='heterollm.cpu_trace.v2',aggregate_capacity=32,
        aggregate_record_bytes=592,histogram='log2_ns_zero_bin',
        detail_policy='complete_except_aggregated_event_calls',aggregate_kinds=['event.record_submit','event.synchronize'])
    rows=[r for r in rows if r.get('kind')!='event.synchronize']
    for i,r in enumerate(rows[1:-1]):r['seq']=i
    rows[-1].update(recorded=len(rows)-2,aggregate_groups=0,aggregate_started=0,aggregate_completed=0,
                    aggregate_unfinished=0,aggregate_dropped=0,counter_overflow=False,invalid_metadata=False)
    return rows


class RecorderV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        (cls.path/'fixture.cpp').write_text(HARNESS);cls.bins={}
        for name,flags in [('enabled',['-DHETEROLLM_CPU_TRACE=1']),('small',['-DHETEROLLM_CPU_TRACE=1','-DHETEROLLM_TRACE_CAPACITY=32']),('disabled',[])]:
            binary=cls.path/name
            subprocess.run(['c++','-std=c++17','-O2','-pthread',*flags,'-I',str(ROOT/'instrumentation/v2'),str(cls.path/'fixture.cpp'),'-o',str(binary)],check=True,capture_output=True,timeout=30)
            cls.bins[name]=binary

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_case(self,mode,variant='enabled'):
        out=self.path/(self.id().split('.')[-1]+'-'+mode+'.jsonl')
        run=subprocess.run([str(self.bins[variant]),mode],env={**os.environ,'HETEROLLM_TRACE_FILE':str(out)},capture_output=True,text=True,timeout=10)
        raw=out.read_text() if out.exists() else ''
        return run,raw

    def test_100000_concurrent_aggregates_preserve_all_interleaved_details_exactly(self):
        run,raw=self.run_case('mixed');self.assertEqual(run.returncode,0);self.assertEqual(run.stderr,'')
        summary=analyze(raw,run.stderr)
        self.assertEqual(summary['aggregate_calls'],100000)
        self.assertEqual(summary['detailed_events'],900)
        self.assertEqual(summary['host_staged_directions']['CUDA0 -> ROCm0']['copies'],100)
        self.assertEqual(summary['host_staged_directions']['CUDA0 -> ROCm0']['logical_bytes'],1600)
        self.assertIsNone(summary['cpu_wait_union_ns']);self.assertIsNone(summary['gpu_execution_time_ns'])
        for a in summary['aggregates'].values():
            odd=a['kind']=='event.synchronize'
            self.assertEqual(a['count'],25000)
            self.assertEqual(a['total_cpu_ns'],100000 if odd else 75000)
            self.assertEqual(a['max_cpu_ns'],7 if odd else 6)
            self.assertEqual(a['histogram'][:4],[0,6250,6250,12500] if odd else [6250,0,6250,12500])
        self.assertLess(len(raw),1024*1024)

    def test_zero_host_stage_complete_trace_is_valid_and_waits_not_double_counted(self):
        run,raw=self.run_case('aggregate_only');s=analyze(raw,run.stderr)
        self.assertEqual(s['host_staged_directions'],{});self.assertEqual(s['detailed_events'],0)
        self.assertIsNone(s['cpu_wait_union_ns']);self.assertEqual(s['detailed_backend_wait_union_ns'],0)
        r=fixture_v2();r.insert(-1,dict(r[2],seq=len(r)-2,begin_ns=3,end_ns=30))
        r[-1]['recorded']+=1
        self.assertEqual(analyze('\n'.join(map(json.dumps,r)),'')['detailed_backend_wait_union_ns'],29)

    def test_loss_unfinished_malformed_and_counter_overflow_fail_closed(self):
        for mode in ('unfinished','detail_unfinished','aggregate_overflow','counter_overflow','metadata','bad_kind','critical_overflow','write_failure'):
            with self.subTest(mode=mode):
                run,raw=self.run_case(mode,'small' if mode=='critical_overflow' else 'enabled')
                self.assertIn('HETEROLLM_TRACE_ERROR',run.stderr)
                with self.assertRaises(ValueError):analyze(raw,run.stderr)
                with self.assertRaises(ValueError):analyze(raw,'')

    def test_disabled_arguments_and_output_unchanged(self):
        run,raw=self.run_case('disabled','disabled');self.assertEqual(run.returncode,0);self.assertEqual(raw,'')

    def test_abrupt_exit_and_existing_output_refused(self):
        run,raw=self.run_case('abrupt');self.assertEqual(run.returncode,7)
        with self.assertRaises(ValueError):analyze(raw,run.stderr)
        first=self.run_case('aggregate_only');again=self.run_case('aggregate_only')
        self.assertEqual(first[1],again[1]);self.assertIn('fresh output open failed',again[0].stderr)

    def test_analyzer_rejects_bad_schema_counters_histograms_and_metadata(self):
        run,raw=self.run_case('aggregate_only');original=[json.loads(l) for l in raw.splitlines()]
        for mutation in ('schema','groups','started','counter','kind','backend','count','histogram','total','max','duplicate','dropped','truncated','complete'):
            rows=deepcopy(original)
            if mutation=='schema':rows[0]['schema']='future'
            elif mutation=='groups':rows[-1]['aggregate_groups']+=1
            elif mutation=='started':rows[-1]['aggregate_started']+=1
            elif mutation=='counter':rows[-1]['counter_overflow']=True
            elif mutation=='kind':rows[1]['kind']='copy.host_stage'
            elif mutation=='backend':rows[1]['backend']='x'*32
            elif mutation=='count':rows[1]['count']=True
            elif mutation=='histogram':rows[1]['histogram'][0]+=1
            elif mutation=='total':rows[1]['total_cpu_ns']=-1
            elif mutation=='max':rows[1]['max_cpu_ns']=rows[1]['total_cpu_ns']+1
            elif mutation=='duplicate':rows.insert(-1,deepcopy(rows[1]));rows[-1]['aggregate_groups']+=1
            elif mutation=='dropped':rows[-1]['dropped']=1
            elif mutation=='truncated':rows[-1]['metadata_truncated']=1
            elif mutation=='complete':rows[-1]['complete']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):analyze('\n'.join(map(json.dumps,rows)),'')

    def test_patch_reproducible_and_only_event_capture_strategy_changes(self):
        spec=importlib.util.spec_from_file_location('v2patch',ROOT/'instrumentation/v2/build_patch.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertEqual(mod.make_patch(),(ROOT/'instrumentation/v2/ggml-cpu-trace.patch').read_text())
        original=(ROOT/'runtime-c2-trace/ggml/src/ggml-backend.cpp').read_text()
        changed=original.replace('HT_TRACE_SCOPE("event.synchronize", ggml_backend_dev_name(event->device), "", 0);','HT_TRACE_EVENT("event.synchronize", ggml_backend_dev_name(event->device));').replace('HT_TRACE_SCOPE("event.record_submit", ggml_backend_name(backend), "", 0);','HT_TRACE_EVENT("event.record_submit", ggml_backend_name(backend));')
        self.assertEqual(mod.patched_source((ROOT/'llama.cpp/ggml/src/ggml-backend.cpp').read_text()),changed)

    def test_histogram_total_must_include_one_call_at_the_reported_maximum(self):
        run,raw=self.run_case('aggregate_only');rows=[json.loads(l) for l in raw.splitlines()]
        a=rows[1];a.update(count=2,total_cpu_ns=8,max_cpu_ns=7,histogram=[0]*65);a['histogram'][3]=2
        rows[-1].update(aggregate_started=2,aggregate_completed=2)
        # Two durations in [4,7] totaling 8 must be 4+4; maximum 7 is impossible.
        with self.assertRaisesRegex(ValueError,'inconsistent'):analyze('\n'.join(map(json.dumps,rows)),'')
        a['total_cpu_ns']=11
        self.assertEqual(analyze('\n'.join(map(json.dumps,rows)),'')['aggregate_calls'],2)
