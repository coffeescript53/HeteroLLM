#pragma once
#include "ht3-api.h"
#include "heterollm-cpu-trace.h"
#include <cmath>
// This file is included ONLY by ggml-backend.cpp in ggml-base.
namespace ht3_detail {
constexpr size_t max_records=16384;
struct State {
    std::mutex mutex;
    FILE * file=nullptr;
    bool enabled=false, invalid=false;
    uint64_t current=0, phases=0, starts=0, ends=0, pending=0;
    bool open[max_records]={};
    State() {
        const char * path=std::getenv("HETEROLLM_TIMING_FILE");
        if (!path || !*path) return;
        enabled=true;
        int fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if (fd<0 || !(file=::fdopen(fd,"w"))) {
            if(fd>=0)::close(fd);
            invalid=true;std::fputs("HETEROLLM_TIMING_ERROR: fresh output open\n",stderr);return;
        }
        std::fputs("{\"type\":\"header\",\"schema\":\"heterollm.timing.v3\",\"clock\":\"steady_clock_ns\",\"capacity\":16384,\"pool_pairs_per_backend\":256,\"correlation\":\"cpu_enqueue_to_completion_enclosure\",\"extra_synchronization\":false}\n",file);
        if(std::fflush(file)||::fsync(fd))fail("header write");
    }
    void fail(const char * why) {
        invalid=true;std::fprintf(stderr,"HETEROLLM_TIMING_ERROR: %s\n",why);
    }
    ~State() {
        if(!file)return;
        bool ok=!invalid && !pending && !current && starts==ends && !std::ferror(file);
        std::fprintf(file,"{\"type\":\"footer\",\"phases\":%llu,\"started\":%llu,\"completed\":%llu,\"pending\":%llu,\"active_phase\":%llu,\"invalid\":%s,\"complete\":%s}\n",(unsigned long long)phases,(unsigned long long)starts,(unsigned long long)ends,(unsigned long long)pending,(unsigned long long)current,invalid?"true":"false",ok?"true":"false");
        if(std::ferror(file)||std::fflush(file)||::fsync(::fileno(file)))ok=false;
        if(std::fclose(file))ok=false;
        if(!ok)std::fputs("HETEROLLM_TIMING_ERROR: incomplete or write failure\n",stderr);
    }
};
inline State & state(){static State s;return s;}
}
extern "C" bool ht3_enabled(){return ht3_detail::state().enabled;}
extern "C" uint64_t ht3_now(){return heterollm_trace::now_ns();}
extern "C" void ht3_error(const char * reason){auto &s=ht3_detail::state();std::lock_guard<std::mutex>l(s.mutex);if(s.enabled)s.fail(reason);}
extern "C" uint64_t ht3_phase_begin(const char * name,int repetition){
    auto &s=ht3_detail::state();if(!s.enabled)return 0;
    std::lock_guard<std::mutex>l(s.mutex);
    if(s.current || s.pending || s.phases>=16 || repetition<0){s.fail("phase begin/state");return 0;}
    if(std::strcmp(name,"warmup_pp")&&std::strcmp(name,"warmup_tg")&&std::strcmp(name,"pp512")&&std::strcmp(name,"tg128")&&std::strcmp(name,"depth")){s.fail("phase name");return 0;}
    s.current=++s.phases;
    if(s.file){std::fprintf(s.file,"{\"type\":\"phase_begin\",\"phase\":%llu,\"cpu_ns\":%llu,\"repetition\":%d,\"name\":",(unsigned long long)s.current,(unsigned long long)ht3_now(),repetition);heterollm_trace::json_string(s.file,name);std::fputs("}\n",s.file);}
    return s.current;
}
extern "C" void ht3_phase_end(uint64_t phase){
    auto &s=ht3_detail::state();if(!s.enabled)return;
    std::lock_guard<std::mutex>l(s.mutex);
    if(!phase || phase!=s.current || s.pending){s.fail("phase end/incomplete device capture");return;}
    if(s.file)std::fprintf(s.file,"{\"type\":\"phase_end\",\"phase\":%llu,\"cpu_ns\":%llu}\n",(unsigned long long)phase,(unsigned long long)ht3_now());
    s.current=0;
}
extern "C" bool ht3_begin(ht3_record *r){
    auto &s=ht3_detail::state();if(!s.enabled)return false;
    std::lock_guard<std::mutex>l(s.mutex);
    if(!s.current)return false; // Loading/initialization deliberately outside device coverage.
    if(s.starts>=ht3_detail::max_records){s.fail("device record capacity");return false;}
    r->id=s.starts++;r->phase=s.current;r->graph=heterollm_trace::context().graph;r->split=heterollm_trace::context().split;
    r->thread=uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    s.open[r->id]=true;++s.pending;return true;
}
extern "C" void ht3_complete(const ht3_record *r){
    auto &s=ht3_detail::state();std::lock_guard<std::mutex>l(s.mutex);
    if(r->id>=s.starts || !s.open[r->id] || r->phase!=s.current || !std::isfinite(r->elapsed_ms) || r->elapsed_ms<0 || !(r->start_before<=r->start_after && r->start_after<=r->end_before && r->end_before<=r->end_after && r->end_after<=r->observed)){s.fail("invalid device completion");return;}
    s.open[r->id]=false;--s.pending;++s.ends;
    if(!s.file)return;
    std::fprintf(s.file,"{\"type\":\"device\",\"id\":%llu,\"phase\":%llu,\"graph\":%llu,\"split\":%d,\"thread\":%llu,\"logical_bytes\":%llu,\"start_before\":%llu,\"start_after\":%llu,\"end_before\":%llu,\"end_after\":%llu,\"observed\":%llu,\"elapsed_ms\":%.9g,\"kind\":",(unsigned long long)r->id,(unsigned long long)r->phase,(unsigned long long)r->graph,r->split,(unsigned long long)r->thread,(unsigned long long)r->bytes,(unsigned long long)r->start_before,(unsigned long long)r->start_after,(unsigned long long)r->end_before,(unsigned long long)r->end_after,(unsigned long long)r->observed,r->elapsed_ms);
    heterollm_trace::json_string(s.file,r->kind);std::fputs(",\"backend\":",s.file);heterollm_trace::json_string(s.file,r->backend);std::fputs(",\"stream\":",s.file);heterollm_trace::json_string(s.file,r->stream);std::fputs("}\n",s.file);
}
