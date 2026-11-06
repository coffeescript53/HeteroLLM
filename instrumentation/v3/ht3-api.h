#pragma once
#include "ggml.h"
#include <stdint.h>
// One implementation in ggml-base. Never duplicate a recorder in backend DSOs.
struct ht3_record {
    uint64_t id, phase, graph, thread, bytes;
    int split;
    uint64_t start_before, start_after, end_before, end_after, observed;
    double elapsed_ms;
    const char * kind;
    const char * backend;
    const char * stream;
};
extern "C" {
GGML_API bool ht3_enabled();
GGML_API uint64_t ht3_now();
GGML_API uint64_t ht3_phase_begin(const char * name, int repetition);
GGML_API void ht3_phase_end(uint64_t phase);
GGML_API bool ht3_begin(ht3_record * record);
GGML_API void ht3_complete(const ht3_record * record);
GGML_API void ht3_error(const char * reason);
}
struct ht3_phase_scope {
    uint64_t id;
    ht3_phase_scope(const char * name, int repetition): id(ht3_phase_begin(name,repetition)) {}
    ~ht3_phase_scope() { ht3_phase_end(id); }
    ht3_phase_scope(const ht3_phase_scope &) = delete;
};
