"""Generate candidate patch only; never alter the pinned llama.cpp working tree."""
import difflib
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = "cdb79f2937aa147084a76f949adc0eb760ad1883e0ccc015bb4e0f599c89bd5f"
REVISION = "5e03bdd8700948b9c41c54dd1b00f28a2aebc03f"


def patched_source(original):
    if hashlib.sha256(original.encode()).hexdigest() != BASE_SHA256:
        raise ValueError("pinned ggml-backend.cpp identity mismatch")
    value = original

    def replace(old, new):
        nonlocal value
        if value.count(old) != 1:
            raise ValueError(f"nonunique or missing patch anchor: {old[:100]}")
        value = value.replace(old, new, 1)

    replace('#include "ggml-impl.h"', '#include "ggml-impl.h"\n#include "heterollm-cpu-trace.h"')
    replace('    backend->iface.synchronize(backend);',
            '    HT_TRACE_SCOPE("backend.synchronize", ggml_backend_name(backend), "", 0);\n    backend->iface.synchronize(backend);')
    replace('    return backend->iface.graph_compute(backend, cgraph);',
            '    HT_TRACE_SCOPE("graph.submit", ggml_backend_name(backend), "", 0);\n    return backend->iface.graph_compute(backend, cgraph);')
    replace('    if (dst_buf->iface.cpy_tensor) {\n        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);',
            '    if (dst_buf->iface.cpy_tensor) {\n        HT_TRACE_SCOPE("copy.buffer_interface_attempt", ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst_buf), ggml_nbytes(src));\n        const bool copied = dst_buf->iface.cpy_tensor(dst_buf, src, dst);\n        if (copied) {\n            HT_TRACE_SCOPE("copy.buffer_interface_accepted", ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst_buf), ggml_nbytes(src));\n        }\n        return copied;')
    replace('    event->device->iface.event_synchronize(event->device, event);',
            '    HT_TRACE_SCOPE("event.synchronize", ggml_backend_dev_name(event->device), "", 0);\n    event->device->iface.event_synchronize(event->device, event);')
    replace('    backend->iface.event_wait(backend, event);',
            '    HT_TRACE_SCOPE("event.wait_submit", ggml_backend_dev_name(event->device), ggml_backend_name(backend), 0);\n    backend->iface.event_wait(backend, event);')
    replace('    backend->iface.event_record(backend, event);',
            '    HT_TRACE_SCOPE("event.record_submit", ggml_backend_name(backend), "", 0);\n    backend->iface.event_record(backend, event);')
    replace('    if (ggml_backend_buffer_is_host(src->buffer)) {\n        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));',
            '    if (ggml_backend_buffer_is_host(src->buffer)) {\n        HT_TRACE_SCOPE("copy.host_to_backend", ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer), ggml_nbytes(src));\n        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));')
    replace('    } else if (ggml_backend_buffer_is_host(dst->buffer)) {\n        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));',
            '    } else if (ggml_backend_buffer_is_host(dst->buffer)) {\n        HT_TRACE_SCOPE("copy.backend_to_host", ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer), ggml_nbytes(src));\n        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));')
    replace('    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {',
            '    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {\n        HT_TRACE_SCOPE("copy.host_stage", ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer), ggml_nbytes(src));')
    replace('        ggml_backend_tensor_get(src, data, 0, nbytes);\n        ggml_backend_tensor_set(dst, data, 0, nbytes);',
            '        {\n            HT_TRACE_SCOPE("copy.host_stage.get", ggml_backend_buffer_name(src->buffer), "host_malloc", nbytes);\n            ggml_backend_tensor_get(src, data, 0, nbytes);\n        }\n        {\n            HT_TRACE_SCOPE("copy.host_stage.set", "host_malloc", ggml_backend_buffer_name(dst->buffer), nbytes);\n            ggml_backend_tensor_set(dst, data, 0, nbytes);\n        }')
    replace('    // an async copy would normally happen after all the queued operations on both backends are completed',
            '    HT_TRACE_SCOPE("copy.public_async_fallback", ggml_backend_name(backend_src), ggml_backend_name(backend_dst), ggml_nbytes(src));\n    // an async copy would normally happen after all the queued operations on both backends are completed')
    replace('void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {',
            'void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {\n    HT_TRACE_SCOPE("graph.split_build", "", "", 0);')
    replace('static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {',
            'static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {\n    HT_TRACE_SCOPE("graph.allocate", "", "", 0);')
    replace('    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {',
            '    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {\n        HT_TRACE_SCOPE("graph.reallocate", backend_ids_changed ? "backend_changed" : "allocation_retry", "", 0);')
    replace('static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {\n    GGML_ASSERT(sched);',
            'static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {\n    GGML_ASSERT(sched);\n    HT_TRACE_GRAPH(sched->n_copies, sched->cur_copy);\n    HT_TRACE_SCOPE("graph.compute_splits", "", "", 0);')
    replace('    for (int split_id = 0; split_id < sched->n_splits; split_id++) {\n        struct ggml_backend_sched_split * split = &splits[split_id];',
            '    for (int split_id = 0; split_id < sched->n_splits; split_id++) {\n        HT_TRACE_SPLIT(split_id);\n        struct ggml_backend_sched_split * split = &splits[split_id];')
    replace('        ggml_backend_t split_backend = sched->backends[split_backend_id];\n\n        // ensure',
            '        ggml_backend_t split_backend = sched->backends[split_backend_id];\n        HT_TRACE_SCOPE("split.cpu_scope", ggml_backend_name(split_backend), "", 0);\n\n        // ensure')
    replace('                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {\n                        ggml_backend_synchronize(input_backend);',
            '                    bool ht_async_copied;\n                    {\n                        HT_TRACE_SCOPE("copy.scheduler_async_attempt", ggml_backend_name(input_backend), ggml_backend_name(split_backend), ggml_nbytes(input));\n                        ht_async_copied = split_backend->iface.cpy_tensor_async && split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy);\n                    }\n                    if (!ht_async_copied) {\n                        HT_TRACE_SCOPE("copy.scheduler_fallback", ggml_backend_name(input_backend), ggml_backend_name(split_backend), ggml_nbytes(input));\n                        ggml_backend_synchronize(input_backend);')
    replace('                        ggml_backend_tensor_copy(input, input_cpy);\n                    }\n                }\n            }\n        }\n\n        if (!sched->callback_eval)',
            '                        ggml_backend_tensor_copy(input, input_cpy);\n                    } else {\n                        HT_TRACE_SCOPE("copy.scheduler_async_accepted", ggml_backend_name(input_backend), ggml_backend_name(split_backend), ggml_nbytes(input));\n                    }\n                }\n            }\n        }\n\n        if (!sched->callback_eval)')
    return value


def make_patch():
    source = (ROOT / "llama.cpp/ggml/src/ggml-backend.cpp").read_text()
    header = (ROOT / "instrumentation/heterollm-cpu-trace.h").read_text()
    patch = ''.join(difflib.unified_diff(source.splitlines(True), patched_source(source).splitlines(True),
                                       "a/ggml/src/ggml-backend.cpp", "b/ggml/src/ggml-backend.cpp"))
    patch += ''.join(difflib.unified_diff([], header.splitlines(True), "/dev/null", "b/ggml/src/heterollm-cpu-trace.h"))
    return patch


if __name__ == "__main__":
    (ROOT / "instrumentation/ggml-cpu-trace.patch").write_text(make_patch())
