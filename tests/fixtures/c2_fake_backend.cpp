// CPU mocks around the ACTUAL patched scheduler/copy implementation. No GPU API.
#include <cstring>
#include <string>
#include <vector>
#ifdef HETEROLLM_LIBRARY_PROBE
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#else
#include HETEROLLM_PATCHED_SOURCE
#endif
#include <iostream>
#include <stdexcept>

static std::vector<std::string> calls;
static bool accept_async=false, accept_buffer=false;
static unsigned char source_bytes[16], destination_bytes[16];
static const char * bname(ggml_backend_t b) { return (const char *) b->context; }
static const char * tname(ggml_backend_buffer_type_t b) { return (const char *) b->context; }
static const char * dname(ggml_backend_dev_t d) { return (const char *) d->context; }
static void sync(ggml_backend_t b) { calls.push_back(std::string("sync:")+bname(b)); }
static bool host(ggml_backend_buffer_type_t b) { return std::string(tname(b))=="HOST"; }
static void get(ggml_backend_buffer_t, const ggml_tensor * t, void * p, size_t o,size_t n) {
    calls.push_back("get"); std::memcpy(p,(char *)t->data+o,n);
}
static void set(ggml_backend_buffer_t, ggml_tensor * t, const void * p,size_t o,size_t n) {
    calls.push_back("set"); std::memcpy((char *)t->data+o,p,n);
}
static bool copy_buffer(ggml_backend_buffer_t,const ggml_tensor * a,ggml_tensor * b) {
    calls.push_back("buffer_try"); if(accept_buffer)std::memcpy(b->data,a->data,ggml_nbytes(a)); return accept_buffer;
}
static bool copy_async(ggml_backend_t,ggml_backend_t,const ggml_tensor * a,ggml_tensor * b) {
    calls.push_back("async_try"); if(accept_async)std::memcpy(b->data,a->data,ggml_nbytes(a)); return accept_async;
}
static void event_sync(ggml_backend_dev_t,ggml_backend_event_t) { calls.push_back("event_sync"); }
static void event_wait(ggml_backend_t,ggml_backend_event_t) { calls.push_back("event_wait"); }
static void event_record(ggml_backend_t,ggml_backend_event_t) { calls.push_back("event_record"); }
static ggml_status compute(ggml_backend_t,ggml_cgraph *) {
    // Fail if the actual scheduler submits before the mock copy completes.
    if(std::memcmp(source_bytes,destination_bytes,16))throw std::runtime_error("submit before copy");
    calls.push_back("graph"); return GGML_STATUS_SUCCESS;
}

int main(int argc,char ** argv) {
    if(argc!=2)return 2;
    const std::string mode=argv[1];
    for(int i=0;i<16;i++)source_bytes[i]=(unsigned char)(i*13+7);
    ggml_backend_buffer_type at={},bt={};
    at.context=(void *)(mode=="host_source"?"HOST":"CUDA0");
    bt.context=(void *)(mode=="host_destination"?"HOST":"ROCm0");
    at.iface.get_name=bt.iface.get_name=tname;at.iface.is_host=bt.iface.is_host=host;
    ggml_backend_buffer ab={},bb={};ab.buft=&at;bb.buft=&bt;ab.size=bb.size=16;
    ab.iface.get_tensor=get;bb.iface.set_tensor=set;bb.iface.cpy_tensor=copy_buffer;
    ggml_tensor a={},b={};a.type=b.type=GGML_TYPE_F32;
    for(int i=0;i<4;i++){a.ne[i]=b.ne[i]=(i==0?4:1);a.nb[i]=b.nb[i]=(i==0?4:16);}
    a.data=source_bytes;b.data=destination_bytes;a.buffer=&ab;b.buffer=&bb;
    ggml_backend aa={},ba={};aa.context=(void *)"CUDA0";ba.context=(void *)"ROCm0";
    aa.iface.get_name=ba.iface.get_name=bname;aa.iface.synchronize=ba.iface.synchronize=sync;
    ba.iface.cpy_tensor_async=copy_async;ba.iface.graph_compute=compute;
    ba.iface.event_wait=event_wait;ba.iface.event_record=event_record;
    ggml_backend_device dev={};dev.context=(void *)"ROCm0";dev.iface.get_name=dname;dev.iface.event_synchronize=event_sync;
    ggml_backend_event event={&dev,nullptr};
    accept_async=mode=="accepted"||mode=="public_accepted";
    accept_buffer=mode=="buffer_accepted";
    if(mode=="flood") {
        // 90,000 genuine low-level calls interleaved with real wrapper copies.
        // The original, trace-disabled and enabled translation units must agree.
        for(int i=0;i<45000;i++) {
            ggml_backend_event_record(&event,&ba);
            ggml_backend_event_synchronize(&event);
            if(i%5000==0) {
                std::memset(destination_bytes,0,16);
                ggml_backend_tensor_copy_async(&aa,&ba,&a,&b);
                if(ggml_backend_graph_compute_async(&ba,nullptr)!=GGML_STATUS_SUCCESS)return 8;
            }
        }
    }
    if(mode=="missing_callback")ba.iface.cpy_tensor_async=nullptr;
    if(mode=="same_tensor") {
        ggml_backend_tensor_copy(&a,&a);ggml_backend_tensor_copy_async(&aa,&ba,&a,&a);
        if(!calls.empty())return 3;
    } else if(mode.rfind("public_",0)==0) {
        ggml_backend_tensor_copy_async(&aa,&ba,&a,&b);
    } else if(mode=="host_source"||mode=="host_destination"||mode=="buffer_accepted") {
        ggml_backend_tensor_copy(&a,&b);
    } else {
#ifdef HETEROLLM_LIBRARY_PROBE
        return 6; // The private scheduler is covered by the translation-unit tests.
#else
        ggml_backend_sched state={};auto * sched=&state;
        sched->n_backends=2;sched->n_copies=1;sched->backends[0]=&aa;sched->backends[1]=&ba;
        sched->hash_set=ggml_hash_set_new(4);
        std::vector<int> backend_ids(sched->hash_set.size,0);
        std::vector<ggml_tensor *> copies(sched->hash_set.size*2,nullptr);
        sched->hv_tensor_backend_ids=backend_ids.data();sched->hv_tensor_copies=copies.data();
        tensor_copy(&a,1,0)=&b;
        ggml_tensor * inputs[]={&a};ggml_tensor * nodes[]={&b};
        ggml_backend_sched_split split={};split.backend_id=1;split.inputs=inputs;split.n_inputs=1;
        split.graph.nodes=nodes;split.graph.n_nodes=1;
        sched->splits=&split;sched->n_splits=1;
        if(mode!="no_event")sched->events[1][0]=&event;
        if(mode=="user_input")a.flags=GGML_TENSOR_FLAG_INPUT;
        if(ggml_backend_sched_compute_splits(sched)!=GGML_STATUS_SUCCESS)return 4;
        ggml_hash_set_free(&sched->hash_set);
#endif
    }
    if(mode!="same_tensor"&&std::memcmp(source_bytes,destination_bytes,16))return 5;
    for(const auto & call:calls)std::cout<<call<<"\n";
}
