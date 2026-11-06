#pragma once
#include "ht3-pool.h"
// Included after common.cuh: each DSO uses ONLY its native event API.
namespace {
struct ht3_ops {
    using Event=cudaEvent_t;using Stream=cudaStream_t;
    static bool create(Event &e){return cudaEventCreateWithFlags(&e,0)==cudaSuccess;}
    static bool record(Event e,Stream s){return cudaEventRecord(e,s)==cudaSuccess;}
    static bool destroy(Event e){return cudaEventDestroy(e)==cudaSuccess;}
    static int query(Event e){
#ifdef GGML_USE_HIP
        auto r=hipEventQuery(e);return r==hipSuccess?1:r==hipErrorNotReady?0:-1;
#else
        auto r=cudaEventQuery(e);return r==cudaSuccess?1:r==cudaErrorNotReady?0:-1;
#endif
    }
    static bool elapsed(float &ms,Event a,Event b){
#ifdef GGML_USE_HIP
        return hipEventElapsedTime(&ms,a,b)==hipSuccess;
#else
        return cudaEventElapsedTime(&ms,a,b)==cudaSuccess;
#endif
    }
};
ht3_pool<ht3_ops> &ht3_events(){static ht3_pool<ht3_ops> value;return value;}
size_t ht3_start(int device,cudaStream_t stream,const char *kind,const char *name,uint64_t bytes=0){
    if(!ht3_enabled())return 256;
    if(device!=0 || !stream){ht3_error("unsupported device or NULL stream");return 256;}
#ifdef GGML_USE_HIP
    const char *backend="ROCm0";
#else
    const char *backend="CUDA0";
#endif
    return ht3_events().begin(stream,kind,backend,name,bytes);
}
}
