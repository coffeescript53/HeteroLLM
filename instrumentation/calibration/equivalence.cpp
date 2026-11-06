// Bounded correctness capture only. Does not modify either pinned runtime.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <dlfcn.h>
#include <sys/prctl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * reason) { if (!ok) throw std::runtime_error(reason); }
using begin_fn = uint64_t (*)(const char *, int);
using end_fn = void (*)(uint64_t);
static begin_fn begin_phase;
static end_fn end_phase;
static FILE * raw;
static FILE * metadata;
static int rows = 0;
static void capture(llama_context * ctx, const llama_vocab * vocab, const char * phase, int pos) {
    const int n = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits(ctx);
    require(logits && n == 128256, "logits/vocabulary mismatch");
    int best = 0;
    for (int i = 0; i < n; ++i) {
        require(std::isfinite(logits[i]), "nonfinite logits");
        if (logits[i] > logits[best]) best = i; // exact ties choose lowest token ID
    }
    require(fwrite(logits, sizeof(float), n, raw) == size_t(n), "logit write failed");
    char piece[4096];
    int size = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
    require(size >= 0 && size <= int(sizeof(piece)), "token-piece output failed");
    require(fprintf(metadata, "{\"type\":\"row\",\"row\":%d,\"phase\":\"%s\",\"position\":%d,\"argmax\":%d,\"piece_hex\":\"", rows, phase, pos, best) > 0, "metadata write failed");
    for (int i = 0; i < size; ++i) require(fprintf(metadata, "%02x", (unsigned char)piece[i]) == 2, "piece write failed");
    require(fprintf(metadata, "\"}\n") > 0, "metadata write failed");
    ++rows;
}
static void run(llama_context * ctx, const llama_vocab * vocab, const std::vector<llama_token> & tokens,
                bool pp, bool save, const char * phase) {
    llama_memory_clear(llama_get_memory(ctx), false);
    uint64_t id = begin_phase ? begin_phase(phase, 0) : 0;
    auto * batch = llama_batch_ext_init(ctx);
    require(batch != nullptr, "batch init failed");
    const int step = pp ? 512 : 1;
    for (int offset = 0; offset < int(tokens.size()); offset += step) {
        llama_batch_ext_clear(batch);
        const int count = std::min(step, int(tokens.size()) - offset);
        for (int j = 0; j < count; ++j) {
            int i = llama_batch_ext_add_token(batch, 0, tokens[offset+j]);
            llama_pos pos = offset+j;
            require(i >= 0 && llama_batch_ext_set_pos(batch, i, &pos) &&
                    llama_batch_ext_set_output_logits(batch, i, j == count-1), "batch setup failed");
        }
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "decode failed");
        // Matches benchmark completion points: once after PP, once per TG step.
        llama_synchronize(ctx);
        if (save) capture(ctx, vocab, phase, offset+count-1);
    }
    llama_batch_ext_free(batch);
    if (end_phase) end_phase(id);
}
int main(int argc, char ** argv) {
    try {
        require(argc == 6 && (std::string(argv[1]) == "U" || std::string(argv[1]) == "T"), "usage: U|T model token-fixture output-dir exact-backend-dir");
        require(prctl(PR_SET_NAME, "llama-bench", 0, 0, 0) == 0, "process-name guard failed");
        const bool traced = std::string(argv[1]) == "T";
        const uint16_t endian = 1;
        require(sizeof(float) == 4 && *(const unsigned char *)&endian == 1, "requires little-endian float32");
        begin_phase = (begin_fn)dlsym(RTLD_DEFAULT, "ht3_phase_begin");
        end_phase = (end_fn)dlsym(RTLD_DEFAULT, "ht3_phase_end");
        require(traced ? (begin_phase && end_phase && getenv("HETEROLLM_TRACE_FILE") && getenv("HETEROLLM_TIMING_FILE")) :
                (!begin_phase && !end_phase && !getenv("HETEROLLM_TRACE_FILE") && !getenv("HETEROLLM_TIMING_FILE")), "runtime/trace mode mismatch");
        std::ifstream input(argv[3]);
        std::vector<llama_token> pp(512), tg(128);
        for (auto * seq : {&pp, &tg}) for (auto & token : *seq) require(bool(input >> token) && token >= 0 && token < 128256, "invalid token fixture");
        std::string extra; require(!(input >> extra) && input.eof(), "extra/invalid fixture bytes");
        require(pp[0] == 128000 && tg[0] == 128000, "BOS fixture mismatch");
        const std::string out(argv[4]);
        raw = fopen((out+"/logits.f32").c_str(), "wbx");
        metadata = fopen((out+"/predictions.jsonl").c_str(), "wx");
        require(raw && metadata, "fresh output required");
        require(fprintf(metadata, "{\"type\":\"header\",\"schema\":\"heterollm.equivalence.v1\",\"arm\":\"%s\",\"vocabulary\":128256,\"rows\":129,\"dtype\":\"little-endian-float32\"}\n", argv[1]) > 0, "header write failed");
        // Dynamic backends search executable/cwd by default. This separate
        // harness must explicitly load only the arm's identity-checked build.
        ggml_backend_load_all_from_path(argv[5]);
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("CUDA0"), ggml_backend_dev_by_name("ROCm0"), nullptr};
        require(devices[0] && devices[1], "both physical runtime devices required");
        std::vector<float> split(llama_max_devices(), 0); split[0] = split[1] = 1;
        auto mp = llama_model_default_params();
        mp.devices=devices; mp.n_gpu_layers=999; mp.split_mode=LLAMA_SPLIT_MODE_LAYER;
        mp.main_gpu=0; mp.tensor_split=split.data(); mp.load_mode=LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode=LLAMA_LAZY_MODE_OFF; mp.no_host=false; mp.use_extra_bufts=true;
        auto * model = llama_model_load_from_file(argv[2], mp); // no fitting API
        require(model != nullptr, "model load failed");
        auto * vocab = llama_model_get_vocab(model);
        require(llama_vocab_n_tokens(vocab) == 128256 && llama_vocab_bos(vocab) == 128000, "vocab/BOS mismatch");
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(cpu != nullptr, "CPU backend missing");
        auto reg=ggml_backend_dev_backend_reg(cpu);
        auto pool_new=(decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_new");
        auto pool_free=(decltype(ggml_threadpool_free) *)ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_free");
        require(pool_new && pool_free, "threadpool callbacks missing");
        for (bool prompt : {true, false}) {
            auto cp=llama_context_default_params();
            cp.n_ctx=prompt ? 512 : 128; cp.n_batch=512; cp.n_ubatch=128;
            cp.n_threads=cp.n_threads_batch=16; cp.type_k=cp.type_v=GGML_TYPE_F16;
            cp.offload_kqv=true; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.embeddings=false; cp.op_offload=true; cp.swa_full=false;
            auto * ctx=llama_init_from_model(model,cp); require(ctx != nullptr,"context failed");
            auto tpp=ggml_threadpool_params_default(16); tpp.strict_cpu=false; tpp.poll=50; tpp.prio=GGML_SCHED_PRIO_NORMAL;
            auto * pool=pool_new(&tpp); require(pool != nullptr,"threadpool failed");
            llama_attach_threadpool(ctx,pool,nullptr); llama_set_n_threads(ctx,16,16);
            if (prompt) { run(ctx,vocab,pp,true,false,"warmup_pp"); run(ctx,vocab,pp,true,true,"pp512"); }
            else { run(ctx,vocab,{tg[0]},false,false,"warmup_tg"); run(ctx,vocab,tg,false,true,"tg128"); }
            llama_free(ctx); pool_free(pool);
        }
        llama_model_free(model); llama_backend_free();
        require(rows==129,"incomplete rows");
        require(fflush(raw)==0 && fclose(raw)==0,"logit close failed"); raw=nullptr;
        require(fprintf(metadata,"{\"type\":\"footer\",\"rows\":129,\"complete\":true}\n")>0 && fflush(metadata)==0 && fclose(metadata)==0,"metadata close failed"); metadata=nullptr;
        puts("{\"schema\":\"heterollm.equivalence.exit.v1\",\"complete\":true,\"rows\":129}");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr,"EQUIVALENCE_ERROR: %s\n",e.what());
        return 1; // supervisor retains partial output; no success footer on failure
    }
}
