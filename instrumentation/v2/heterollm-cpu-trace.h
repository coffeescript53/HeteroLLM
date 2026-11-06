// Bounded CPU-call instrumentation only; no added GPU synchronization.
#pragma once
#if defined(HETEROLLM_CPU_TRACE) && HETEROLLM_CPU_TRACE
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

namespace heterollm_trace {
#ifndef HETEROLLM_TRACE_CAPACITY
#define HETEROLLM_TRACE_CAPACITY 65536
#endif
static constexpr size_t capacity = HETEROLLM_TRACE_CAPACITY;
static constexpr size_t aggregate_capacity = 32, histogram_bins = 65;
static_assert(capacity > 0 && capacity <= 65536, "bounded detailed capacity");
struct Context { uint64_t graph = 0; int split = -1, copy = -1, n_copies = 0; };
inline Context & context() { static thread_local Context value; return value; }
struct Event {
    uint64_t seq = 0, thread = 0, begin_ns = 0, end_ns = 0, logical_bytes = 0;
    Context context;
    char kind[48] = {}, src[32] = {}, dst[32] = {};
};
struct Aggregate {
    char backend[32] = {};
    unsigned kind = 0;
    uint64_t count = 0, total_ns = 0, max_ns = 0, pending = 0;
    uint64_t histogram[histogram_bins] = {};
};
inline uint64_t now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline size_t histogram_bin(uint64_t ns) {
    size_t bin = 0;
    while (ns) { ++bin; ns >>= 1; }
    return bin; // 0: exactly 0ns; b>0: [2^(b-1), 2^b-1]ns.
}
inline void json_string(FILE * file, const char * value) {
    std::fputc('"', file);
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') std::fprintf(file, "\\%c", *p);
        else if (*p < 32) std::fprintf(file, "\\u%04x", unsigned(*p));
        else std::fputc(*p, file);
    }
    std::fputc('"', file);
}
class Recorder {
    std::mutex mutex_;
    Event * events_ = nullptr;
    Aggregate aggregates_[aggregate_capacity] = {};
    FILE * file_ = nullptr;
    size_t count_ = 0, groups_ = 0;
    uint64_t dropped_ = 0, truncated_ = 0, graph_ = 0;
    uint64_t aggregate_started_ = 0, aggregate_completed_ = 0, aggregate_dropped_ = 0;
    bool counter_overflow_ = false, invalid_ = false;
    void add(uint64_t & value, uint64_t amount = 1) {
        if (amount > std::numeric_limits<uint64_t>::max() - value) counter_overflow_ = true;
        else value += amount;
    }
    template<size_t N> void copy_name(char (&dst)[N], const char * src) {
        src = src ? src : "";
        const size_t length = std::strlen(src);
        if (length >= N) add(truncated_);
        std::memcpy(dst, src, length < N ? length : N - 1);
    }
public:
    Recorder() {
        const char * path = std::getenv("HETEROLLM_TRACE_FILE");
        if (!path || !*path) return;
        const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) { std::fprintf(stderr, "HETEROLLM_TRACE_ERROR: fresh output open failed\n"); return; }
        file_ = ::fdopen(fd, "w");
        if (!file_) { ::close(fd); std::fprintf(stderr, "HETEROLLM_TRACE_ERROR: fdopen failed\n"); return; }
        events_ = new (std::nothrow) Event[capacity];
        if (!events_) { std::fprintf(stderr, "HETEROLLM_TRACE_ERROR: allocation failed\n"); return; }
        std::fprintf(file_, "{\"type\":\"header\",\"schema\":\"heterollm.cpu_trace.v2\",\"clock\":\"steady_clock_ns\",\"time_domain\":\"cpu\",\"phase\":\"unknown\",\"capacity\":%zu,\"record_bytes\":%zu,\"aggregate_capacity\":%zu,\"aggregate_record_bytes\":%zu,\"histogram\":\"log2_ns_zero_bin\",\"detail_policy\":\"complete_except_aggregated_event_calls\",\"aggregate_kinds\":[\"event.record_submit\",\"event.synchronize\"]}\n", capacity, sizeof(Event), aggregate_capacity, sizeof(Aggregate));
        if (std::fflush(file_) != 0 || ::fsync(fd) != 0) {
            std::fprintf(stderr, "HETEROLLM_TRACE_ERROR: header durability failed\n");
            delete[] events_; events_ = nullptr;
        }
    }
    ~Recorder() {
        if (!file_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t unfinished = 0;
        uint64_t aggregate_unfinished = 0;
        for (size_t i = 0; i < count_; ++i) {
            const Event & e = events_[i];
            if (!e.end_ns) ++unfinished;
            std::fprintf(file_, "{\"type\":\"span\",\"seq\":%llu,\"thread\":%llu,\"begin_ns\":%llu,\"end_ns\":%llu,\"graph\":%llu,\"split\":%d,\"copy\":%d,\"n_copies\":%d,\"logical_bytes\":%llu,\"kind\":",
                (unsigned long long)e.seq, (unsigned long long)e.thread,
                (unsigned long long)e.begin_ns, (unsigned long long)e.end_ns,
                (unsigned long long)e.context.graph, e.context.split, e.context.copy, e.context.n_copies,
                (unsigned long long)e.logical_bytes);
            json_string(file_, e.kind); std::fputs(",\"src\":", file_); json_string(file_, e.src);
            std::fputs(",\"dst\":", file_); json_string(file_, e.dst); std::fputs("}\n", file_);
        }
        for (size_t i = 0; i < groups_; ++i) {
            const Aggregate & a = aggregates_[i]; add(aggregate_unfinished, a.pending);
            std::fputs("{\"type\":\"aggregate\",\"kind\":", file_);
            json_string(file_, a.kind == 0 ? "event.record_submit" : "event.synchronize");
            std::fputs(",\"backend\":", file_); json_string(file_, a.backend);
            std::fprintf(file_, ",\"count\":%llu,\"total_cpu_ns\":%llu,\"max_cpu_ns\":%llu,\"histogram\":[",
                (unsigned long long)a.count, (unsigned long long)a.total_ns, (unsigned long long)a.max_ns);
            for (size_t b = 0; b < histogram_bins; ++b)
                std::fprintf(file_, "%s%llu", b ? "," : "", (unsigned long long)a.histogram[b]);
            std::fputs("]}\n", file_);
        }
        const bool complete = events_ && !dropped_ && !unfinished && !truncated_ &&
            !aggregate_unfinished && !aggregate_dropped_ && !counter_overflow_ && !invalid_ &&
            aggregate_started_ == aggregate_completed_ && !std::ferror(file_);
        std::fprintf(file_, "{\"type\":\"footer\",\"recorded\":%zu,\"dropped\":%llu,\"unfinished\":%zu,\"metadata_truncated\":%llu,\"aggregate_groups\":%zu,\"aggregate_started\":%llu,\"aggregate_completed\":%llu,\"aggregate_unfinished\":%llu,\"aggregate_dropped\":%llu,\"counter_overflow\":%s,\"invalid_metadata\":%s,\"complete\":%s}\n",
            count_, (unsigned long long)dropped_, unfinished, (unsigned long long)truncated_, groups_,
            (unsigned long long)aggregate_started_, (unsigned long long)aggregate_completed_,
            (unsigned long long)aggregate_unfinished, (unsigned long long)aggregate_dropped_,
            counter_overflow_ ? "true" : "false", invalid_ ? "true" : "false", complete ? "true" : "false");
        bool failed = std::fflush(file_) != 0;
        failed = (::fsync(::fileno(file_)) != 0) || failed;
        failed = (std::fclose(file_) != 0) || failed;
        if (failed || !complete) std::fprintf(stderr, "HETEROLLM_TRACE_ERROR: incomplete or unwritable trace\n");
        delete[] events_;
    }
    bool enabled() const { return events_ != nullptr; }
    uint64_t next_graph() { std::lock_guard<std::mutex> lock(mutex_); add(graph_); return graph_; }
    size_t begin(const char * kind, const char * src, const char * dst, uint64_t bytes) {
        if (!events_) return capacity;
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == capacity) { add(dropped_); return capacity; }
        size_t slot = count_++; Event & e = events_[slot]; e.seq = slot;
        e.thread = uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        e.context = context(); e.logical_bytes = bytes;
        copy_name(e.kind, kind); copy_name(e.src, src); copy_name(e.dst, dst);
        e.begin_ns = now_ns(); return slot;
    }
    void end(size_t slot) {
        if (slot == capacity) return;
        const uint64_t ended = now_ns();
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= count_ || events_[slot].end_ns || ended < events_[slot].begin_ns) { invalid_ = true; return; }
        events_[slot].end_ns = ended;
    }
    size_t aggregate_begin(const char * kind, const char * backend) {
        if (!events_) return aggregate_capacity;
        std::lock_guard<std::mutex> lock(mutex_); add(aggregate_started_);
        unsigned k;
        if (kind && !std::strcmp(kind, "event.record_submit")) k = 0;
        else if (kind && !std::strcmp(kind, "event.synchronize")) k = 1;
        else { invalid_ = true; add(aggregate_dropped_); return aggregate_capacity; }
        if (!backend || !*backend || std::strlen(backend) >= sizeof(Aggregate::backend)) {
            invalid_ = true; add(aggregate_dropped_); return aggregate_capacity;
        }
        size_t slot = 0;
        while (slot < groups_ && (aggregates_[slot].kind != k || std::strcmp(aggregates_[slot].backend, backend))) ++slot;
        if (slot == aggregate_capacity) { add(aggregate_dropped_); return aggregate_capacity; }
        if (slot == groups_) { ++groups_; aggregates_[slot].kind = k; copy_name(aggregates_[slot].backend, backend); }
        add(aggregates_[slot].pending); return slot;
    }
    void aggregate_end(size_t slot, uint64_t elapsed) {
        if (slot == aggregate_capacity) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= groups_ || !aggregates_[slot].pending) { invalid_ = true; return; }
        Aggregate & a = aggregates_[slot]; --a.pending;
        add(a.count); add(a.total_ns, elapsed); add(a.histogram[histogram_bin(elapsed)]);
        if (elapsed > a.max_ns) a.max_ns = elapsed;
        add(aggregate_completed_);
    }
};
inline Recorder & recorder() { static Recorder value; return value; }
class Span {
    size_t slot_;
public:
    Span(const char * kind, const char * src, const char * dst, uint64_t bytes = 0)
        : slot_(recorder().begin(kind, src, dst, bytes)) {}
    ~Span() { recorder().end(slot_); }
    Span(const Span &) = delete;
};
class AggregateSpan {
    size_t slot_; uint64_t start_;
public:
    AggregateSpan(const char * kind, const char * backend)
        : slot_(recorder().aggregate_begin(kind, backend)), start_(now_ns()) {}
    ~AggregateSpan() { recorder().aggregate_end(slot_, now_ns() - start_); }
    AggregateSpan(const AggregateSpan &) = delete;
};
class GraphContext {
    Context previous_;
public:
    GraphContext(int n_copies, int copy) : previous_(context()) {
        if (recorder().enabled()) { context().graph = recorder().next_graph(); context().split = -1; context().copy = copy; context().n_copies = n_copies; }
    }
    ~GraphContext() { context() = previous_; }
};
class SplitContext {
    int previous_;
public:
    explicit SplitContext(int split) : previous_(context().split) { context().split = split; }
    ~SplitContext() { context().split = previous_; }
};
} // namespace heterollm_trace
#define HT_TRACE_JOIN_(a,b) a##b
#define HT_TRACE_JOIN(a,b) HT_TRACE_JOIN_(a,b)
#define HT_TRACE_SCOPE(kind,src,dst,bytes) heterollm_trace::Span HT_TRACE_JOIN(ht_trace_,__LINE__)(kind,src,dst,bytes)
#define HT_TRACE_EVENT(kind,backend) heterollm_trace::AggregateSpan HT_TRACE_JOIN(ht_event_,__LINE__)(kind,backend)
#define HT_TRACE_GRAPH(n,c) heterollm_trace::GraphContext HT_TRACE_JOIN(ht_graph_,__LINE__)(n,c)
#define HT_TRACE_SPLIT(s) heterollm_trace::SplitContext HT_TRACE_JOIN(ht_split_,__LINE__)(s)
#else
#define HT_TRACE_SCOPE(kind,src,dst,bytes) ((void)0)
#define HT_TRACE_EVENT(kind,backend) ((void)0)
#define HT_TRACE_GRAPH(n,c) ((void)0)
#define HT_TRACE_SPLIT(s) ((void)0)
#endif
