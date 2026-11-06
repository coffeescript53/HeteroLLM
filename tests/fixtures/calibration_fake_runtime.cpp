// CPU fake C API for the *actual* correctness harness. Never links CUDA/HIP.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
struct llama_model {};
struct llama_vocab {};
struct llama_context { int size; int pos=0; bool pending=false; std::vector<float> logits; };
struct llama_batch_ext { llama_context * ctx; std::vector<int> tokens,positions; std::vector<bool> outputs; };
static int phases=0,contexts=0,clears=0,decoded=0,syncs=0;
static llama_context * active;
static const char * phase_names[]={"warmup_pp","pp512","warmup_tg","tg128"};
#ifdef FAKE_TIMING
extern "C" uint64_t ht3_phase_begin(const char * name,int repetition) {
    assert(phases<4 && std::strcmp(name,phase_names[phases])==0 && repetition==0);
    return ++phases;
}
extern "C" void ht3_phase_end(uint64_t id) { assert(id==uint64_t(phases) && !active->pending); }
#endif
extern "C" {
void ggml_backend_load_all_from_path(const char * path) { assert(std::strstr(path,"/build-mixed/bin") != nullptr); }
void llama_backend_init() {}
void llama_numa_init(ggml_numa_strategy s) { assert(s==GGML_NUMA_STRATEGY_DISABLED); }
llama_model_params llama_model_default_params() { return {}; }
llama_context_params llama_context_default_params() { return {}; }
size_t llama_max_devices() { return 16; }
ggml_backend_dev_t ggml_backend_dev_by_name(const char * s) { return (ggml_backend_dev_t)(uintptr_t)(std::strcmp(s,"CUDA0")==0 ? 1 : 2); }
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type) { return (ggml_backend_dev_t)3; }
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t) { return (ggml_backend_reg_t)4; }
struct ggml_threadpool_params ggml_threadpool_params_default(int n) { ggml_threadpool_params p={};p.n_threads=n;return p; }
static ggml_threadpool_t new_pool(ggml_threadpool_params * p) { assert(p->n_threads==16 && p->poll==50 && !p->strict_cpu); return (ggml_threadpool_t)5; }
static void free_pool(ggml_threadpool_t) {}
void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t,const char * n) { return std::strcmp(n,"ggml_threadpool_new")==0 ? (void*)new_pool : (void*)free_pool; }
llama_model * llama_model_load_from_file(const char *,llama_model_params p) {
    assert(p.n_gpu_layers==999 && p.split_mode==LLAMA_SPLIT_MODE_LAYER && p.load_mode==LLAMA_LOAD_MODE_NONE && p.lazy_mode==LLAMA_LAZY_MODE_OFF && p.main_gpu==0 && p.tensor_split[0]==1 && p.tensor_split[1]==1 && p.use_extra_bufts && !p.no_host && p.devices[2]==nullptr); return new llama_model;
}
const llama_vocab * llama_model_get_vocab(const llama_model *) { static llama_vocab v;return &v; }
int32_t llama_vocab_n_tokens(const llama_vocab *) { return 128256; }
llama_token llama_vocab_bos(const llama_vocab *) { return 128000; }
llama_context * llama_init_from_model(llama_model *,llama_context_params p) {
    assert(p.n_ctx==(contexts++==0 ? 512 : 128) && p.n_batch==512 && p.n_ubatch==128 && p.n_threads==16 && p.n_threads_batch==16 && p.offload_kqv && p.op_offload && p.type_k==GGML_TYPE_F16 && p.type_v==GGML_TYPE_F16 && p.flash_attn_type==LLAMA_FLASH_ATTN_TYPE_DISABLED && !p.embeddings && !p.swa_full);
    return active=new llama_context{int(p.n_ctx),0,false,std::vector<float>(128256)};
}
void llama_attach_threadpool(llama_context *,ggml_threadpool_t,ggml_threadpool_t) {}
void llama_set_n_threads(llama_context *,int n,int b) { assert(n==16 && b==16); }
llama_memory_t llama_get_memory(const llama_context * c) { return (llama_memory_t)c; }
void llama_memory_clear(llama_memory_t m,bool data) { auto * c=(llama_context*)m;assert(!data && !c->pending);c->pos=0;++clears; }
llama_batch_ext * llama_batch_ext_init(llama_context * c) { return new llama_batch_ext{c,{},{},{}}; }
void llama_batch_ext_free(llama_batch_ext * b) { delete b; }
void llama_batch_ext_clear(llama_batch_ext * b) { b->tokens.clear();b->positions.clear();b->outputs.clear(); }
int32_t llama_batch_ext_add_token(llama_batch_ext * b,llama_seq_id seq,llama_token token) { assert(seq==0);b->tokens.push_back(token);b->positions.push_back(-1);b->outputs.push_back(false);return int(b->tokens.size()-1); }
bool llama_batch_ext_set_pos(llama_batch_ext * b,int32_t i,const llama_pos * p) { b->positions[i]=*p;return true; }
bool llama_batch_ext_set_output_logits(llama_batch_ext * b,int32_t i,bool v) { b->outputs[i]=v;return true; }
int32_t llama_process(llama_context * c,llama_process_type type,llama_batch_ext * b) {
    if(getenv("FAKE_DECODE_ERROR"))return -1;
    assert(type==LLAMA_PROCESS_TYPE_DECODE && !c->pending);
    for(size_t i=0;i<b->tokens.size();++i) {
        int pos=c->pos++;
        int expected=pos==0 ? 128000 : 32+(((pos-1)*(c->size==512?1543:2017)+(c->size==512?17:29))%4064);
        assert(b->tokens[i]==expected && b->positions[i]==pos && b->outputs[i]==(i==b->tokens.size()-1));
    }
    assert(b->tokens.size()==(c->size==512 ? 512 : 1));
    int best=c->pos%128256;
    std::fill(c->logits.begin(),c->logits.end(),-2.f);
    c->logits[best]=3.f;
    if(getenv("FAKE_NONFINITE"))c->logits[0]=std::numeric_limits<float>::quiet_NaN();
    c->pending=true;++decoded;return 0;
}
void llama_synchronize(llama_context * c) { assert(c->pending);c->pending=false;++syncs; }
float * llama_get_logits(llama_context * c) { assert(!c->pending);return c->logits.data(); }
int32_t llama_token_to_piece(const llama_vocab *,llama_token t,char * out,int32_t n,int32_t lstrip,bool special) { assert(lstrip==0 && special);return snprintf(out,n,"token:%d",t); }
void llama_free(llama_context * c) { assert(!c->pending);delete c; }
void llama_model_free(llama_model * m) { delete m; }
void llama_backend_free() { assert(contexts==2 && clears==4 && decoded==131 && syncs==131);
#ifdef FAKE_TIMING
 assert(phases==4);
#endif
}
}
