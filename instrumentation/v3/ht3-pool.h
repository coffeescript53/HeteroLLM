#pragma once
#include "ht3-api.h"
#include <cmath>
#include <mutex>
// Ops is CUDA/HIP in the candidate and deterministic CPU callbacks in tests.
// Only query; NEVER synchronize, wait, poll, or change stream dependencies here.
template<class Ops, size_t Capacity=256> class ht3_pool {
    struct Slot {typename Ops::Event start{},end{};bool start_created=false,end_created=false,pending=false,ended=false;ht3_record row{};};
    Slot slots[Capacity];std::mutex mutex;
public:
    size_t begin(typename Ops::Stream stream,const char *kind,const char *backend,const char *stream_name,uint64_t bytes){
        if(!ht3_enabled())return Capacity;
        std::lock_guard<std::mutex>l(mutex);
        ht3_record r{};r.kind=kind;r.backend=backend;r.stream=stream_name;r.bytes=bytes;
        if(!ht3_begin(&r))return Capacity;
        size_t i=0;while(i<Capacity && slots[i].pending)++i;
        if(i==Capacity){ht3_error("event pool capacity");return Capacity;}
        auto &s=slots[i];s.row=r;s.pending=true;s.ended=false;
        if(!s.start_created)s.start_created=Ops::create(s.start);
        if(!s.end_created)s.end_created=Ops::create(s.end);
        if(!s.start_created||!s.end_created){ht3_error("event create");return Capacity;}
        s.row.start_before=ht3_now();bool ok=Ops::record(s.start,stream);s.row.start_after=ht3_now();
        if(!ok){ht3_error("start record");return Capacity;}return i;
    }
    void end(size_t i,typename Ops::Stream stream){
        if(i==Capacity)return;
        std::lock_guard<std::mutex>l(mutex);auto&s=slots[i];
        if(!s.pending||s.ended){ht3_error("duplicate end");return;}
        s.row.end_before=ht3_now();bool ok=Ops::record(s.end,stream);s.row.end_after=ht3_now();
        s.ended=true;if(!ok)ht3_error("end record");
    }
    void resolve(){
        if(!ht3_enabled())return;
        std::lock_guard<std::mutex>l(mutex);
        for(auto&s:slots){
            if(!s.pending||!s.ended)continue;
            int a=Ops::query(s.start),b=Ops::query(s.end);
            uint64_t observed=ht3_now();
            if(a<0||b<0){ht3_error("event query");continue;}
            if(!a||!b)continue; // Incomplete is retained and fails phase/final validation.
            float ms=0;
            if(!Ops::elapsed(ms,s.start,s.end)||!std::isfinite(ms)||ms<0){ht3_error("elapsed time");continue;}
            s.row.elapsed_ms=ms;s.row.observed=observed;ht3_complete(&s.row);s.pending=false;
        }
    }
    // Called before backend context destruction, after existing workload waits.
    void release(){
        if(!ht3_enabled())return;
        std::lock_guard<std::mutex>l(mutex);
        for(auto&s:slots){
            if(s.pending)ht3_error("unresolved event at release");
            if(s.start_created){if(!Ops::destroy(s.start))ht3_error("start event destroy");s.start_created=false;}
            if(s.end_created){if(!Ops::destroy(s.end))ht3_error("end event destroy");s.end_created=false;}
        }
    }
};
