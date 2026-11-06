// Deterministic CPU-only event adapter: never links/initializes CUDA or HIP.
#include "ht3-recorder.h"
#include "ht3-pool.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
static std::string mode;
static std::vector<std::string> original, events;
static bool completed=false;
struct Fake {
    using Event=int;using Stream=int;
    static bool create(Event&e){static int id=0;e=++id;events.push_back("create");return mode!="create_failure";}
    static bool destroy(Event){events.push_back("destroy");return mode!="destroy_failure";}
    static bool record(Event,Stream){events.push_back("record");return mode!="record_failure";}
    static int query(Event){events.push_back("query");return mode=="query_failure"?-1:completed && mode!="not_ready";}
    static bool elapsed(float&ms,Event,Event){events.push_back("elapsed");ms=mode=="nan"?std::numeric_limits<float>::quiet_NaN():0.000001f;return mode!="elapsed_failure";}
};
int main(int argc,char**argv){
    mode=argc>1?argv[1]:"valid";
    ht3_pool<Fake,2> pool;
    const char *names[]={"warmup_pp","pp512","warmup_tg","tg128"};
    for(const char *name:names){
        ht3_phase_scope phase(name,0);
        heterollm_trace::GraphContext graph(1,0);heterollm_trace::SplitContext split(1);
        if(mode=="record_overflow"){
            for(int i=0;i<16385;i++){
                auto id=pool.begin(1,"graph_uncaptured","CUDA0","main",0);
                pool.end(id,1);completed=true;pool.resolve();
            }
            break;
        }
        for(const char *backend:{"CUDA0","ROCm0"}){
            completed=false;
            size_t id;
            {heterollm_trace::Span submit("graph.submit",backend,"",0);
             id=pool.begin(1,"graph_uncaptured",backend,"main",0);original.push_back("graph");pool.end(id,1);}
            if(mode=="overflow"){
                pool.begin(1,"graph_uncaptured",backend,"main",0);pool.begin(1,"graph_uncaptured",backend,"main",0);
            }
            pool.resolve(); // Not ready: no additional wait or elapsed query allowed.
            original.push_back("existing_sync");completed=true;pool.resolve();
        }
        if(mode!="zero"){
            unsigned char a[16],host[16]={},b[16]={};for(int i=0;i<16;i++)a[i]=i;
            heterollm_trace::Span parent("copy.host_stage","CUDA0","ROCm0",16);
            for(int leg=0;leg<2;leg++){
                heterollm_trace::Span scope(leg?"copy.host_stage.set":"copy.host_stage.get",leg?"host_malloc":"CUDA0",leg?"ROCm0":"host_malloc",16);
                completed=false;
                auto id=pool.begin(2,leg?"copy_h2d":"copy_d2h",leg?"ROCm0":"CUDA0","per_thread",16);
                original.push_back(leg?"set":"get");std::memcpy(leg?b:host,leg?host:a,16);
                pool.end(id,2);original.push_back("existing_sync");completed=true;pool.resolve();
            }
            assert(std::memcmp(a,b,16)==0);
        }
        if(mode=="abrupt")std::_Exit(0);
    }
    pool.release();
    if(mode=="write_failure")assert(std::freopen("/dev/full","w",ht3_detail::state().file));
    for(auto&s:original)std::cout<<s<<"\n";
    if(ht3_enabled() && mode=="valid"){
        assert(std::count(events.begin(),events.end(),"elapsed")==16);
        assert(std::count(events.begin(),events.end(),"record")==32);
        assert(std::count(events.begin(),events.end(),"create")==2);
        assert(std::count(events.begin(),events.end(),"destroy")==2);
    }
    if(!ht3_enabled())assert(events.empty());
}
