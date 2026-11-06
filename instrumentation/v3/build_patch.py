"""Deterministic pinned-source patch; original/v1/v2 trees are read-only."""
import difflib,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('v2patch',ROOT/'instrumentation/v2/build_patch.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
FILES=['ggml/src/ggml-backend.cpp','ggml/src/ggml-cuda/ggml-cuda.cu','tools/llama-bench/llama-bench.cpp']
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('patch anchor: '+a[:100])
    return s.replace(a,b,1)
def patched():
    values={p:(ROOT/'llama.cpp'/p).read_text() for p in FILES}
    p=FILES[0];values[p]=v2.patched_source(values[p])+'\n#include "ht3-recorder.h"\n'
    p=FILES[1];s=values[p]
    s=once(s,'#include "ggml-cuda/common.cuh"','#include "ggml-cuda/common.cuh"\n#include "ht3-cuda.h"')
    # Blocking buffer copies: event END precedes the original wait; no new wait.
    for name,kind,size in [('set_tensor','copy_h2d','size'),('get_tensor','copy_d2h','size'),('set_tensor_2d','copy_h2d_2d','size*n_copies'),('get_tensor_2d','copy_d2h_2d','size*n_copies')]:
        start=s.index('static void ggml_backend_cuda_buffer_'+name+'(');end=s.index('\n}',start)+2
        fn=s[start:end]
        fn=once(fn,'    ggml_cuda_set_device(ctx->device);','    ggml_cuda_set_device(ctx->device);\n    const auto ht_event=ht3_start(ctx->device,cudaStreamPerThread,"'+kind+'","per_thread",'+size+');')
        fn=once(fn,'    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));','    ht3_events().end(ht_event,cudaStreamPerThread);\n    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));\n    ht3_events().resolve();')
        s=s[:start]+fn+s[end:]
    s=once(s,'    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));\n\n    GGML_UNUSED(backend);','    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));\n    ht3_events().resolve();\n\n    GGML_UNUSED(backend);')
    s=once(s,'    delete cuda_ctx;\n    delete backend;','    if(ht3_enabled()){\n        ggml_cuda_set_device(cuda_ctx->device);\n        ht3_events().release();\n    }\n    delete cuda_ctx;\n    delete backend;')
    s=once(s,'        // Launch graph\n        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));','        // Timing markers are outside capture and surround replay on stream0.\n        const auto ht_event=ht3_start(cuda_ctx->device,cuda_ctx->stream(),"graph_replay","main");\n        // Launch graph\n        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));\n        ht3_events().end(ht_event,cuda_ctx->stream());')
    s=once(s,'    if (use_cuda_graph && cuda_graph_update_required) {\n        // Start CUDA graph capture','    const auto ht_event=!use_cuda_graph ? ht3_start(cuda_ctx->device,cuda_ctx->stream(),"graph_uncaptured","main") : 256;\n    if(ht3_enabled() && cuda_ctx->curr_stream_no!=0)ht3_error("graph start outside main stream");\n    if (use_cuda_graph && cuda_graph_update_required) {\n        // Start CUDA graph capture')
    s=once(s,'    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);','    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);\n    if(ht3_enabled() && cuda_ctx->curr_stream_no!=0)ht3_error("graph end outside main stream");\n    if(!use_cuda_graph)ht3_events().end(ht_event,cuda_ctx->stream());')
    s=once(s,'#ifdef USE_CUDA_GRAPH\n        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);','#ifdef USE_CUDA_GRAPH\n        if(ht3_enabled() && is_concurrent_event_active)ht3_error("unjoined auxiliary stream");\n        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);')
    values[p]=s
    p=FILES[2];s=values[p]
    s='#include "../../ggml/src/ht3-api.h"\n'+s
    # Wrap exact calls, never move or replace benchmark timers or existing waits.
    for call,name,rep in [('test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads)','warmup_pp','0'),('test_gen(ctx, 1, t.n_threads)','warmup_tg','0'),('test_prompt(ctx, t.n_depth, t.n_batch, t.n_threads)','depth','i'),('test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads)','pp512','i'),('test_gen(ctx, t.n_gen, t.n_threads)','tg128','i')]:
        old='bool res = '+call+';'
        # first occurrence is warmup, then measured prompt.
        index=s.index(old)
        new='bool res;\n                { ht3_phase_scope ht_phase("'+name+'",'+rep+'); res = '+call+'; }'
        s=s[:index]+new+s[index+len(old):]
    values[p]=s
    values['ggml/src/heterollm-cpu-trace.h']=(ROOT/'instrumentation/v2/heterollm-cpu-trace.h').read_text()
    for name in ('ht3-api.h','ht3-recorder.h','ht3-pool.h','ht3-cuda.h'):values['ggml/src/'+name]=(ROOT/'instrumentation/v3'/name).read_text()
    return values
def make_patch():
    result=''
    for name,value in patched().items():
        p=ROOT/'llama.cpp'/name;old=p.read_text() if p.exists() else ''
        result+=''.join(difflib.unified_diff(old.splitlines(True),value.splitlines(True),'a/'+name if p.exists() else '/dev/null','b/'+name))
    return result
if __name__=='__main__':(ROOT/'instrumentation/v3/phase-timing.patch').write_text(make_patch())
