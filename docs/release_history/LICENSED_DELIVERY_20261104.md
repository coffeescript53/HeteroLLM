# Release preparation complete

Prepared 4 November 2026 from the verified 119-file C4 public candidate. Research status is current through November 3. This repository is prepared with MIT for original contributions and separate upstream/font notices, awaiting the owner's final publication decision; it has not been published.

## What is ready

- Concise README centered on the demonstrated recorder repair and controlled baseline measurements.
- Preserved Python/C++ source, versioned instrumentation, architecture diagram and editable source, three measurement charts, original timing extracts and provenance.
- Linux CPU-only quickstart, optional standalone C++ recorder tests, installation guidance, frozen build/workload specifications and explicit prerequisite boundaries.
- Clear original-versus-upstream attribution, engineering failure analysis, resume bullets and interview explanations.
- Current C5 status: strict numerical comparison FAILED, exploratory quality incomplete after the R3 thermal stop, and controlled performance comparison BLOCKED.

The public source remains the tested C4 snapshot. Later host-specific C5/R3 implementations are not copied from the private workspace. No production algorithm, runtime, safety limit, model, chart or accepted measurement was changed. No GPU experiment occurred.

## Validation

**205 unique CPU tests passed; one expected private-archive skip; no unresolved failure.**

The selected standard-library suite contains 199 tests including the C3 synthetic checks: 198 pass and one explicitly unavailable original-archive comparison. Seven additional standalone C++17 recorder tests pass, including 100,000 concurrent aggregate events interleaved with 900 detailed spans. These are synthetic tests, not GPU performance observations.

The November 4 final license audit reran both documented CPU entrypoints successfully. The earlier preparation used an initial run and focused correction replay. The initial run exposed two public test assertions still bound to original private fixture hashes after C4's documented path sanitization. Only those public digest expectations/comments were corrected; fixture bytes and historical hashes remain preserved. The fake-runtime executable permission lost in ZIP extraction was restored. CLI help, a device-free synthetic snapshot preview, and shell syntax checks also passed.

An offline wheel build passed without installation or network access and included all four license/notice files byte-for-byte. Packaging now uses standard SPDX metadata with setuptools >=77; no installed dependency was upgraded.

Not performed: private host-contract replay, native patch reconstruction requiring omitted upstream sources, package installation, chart regeneration, GPU execution or hardware probing. Installation and hardware reproduction prerequisites are explicit in [REPRODUCE.md](docs/REPRODUCE.md).

## Audit and preservation

The release was inspected for recognizable credentials, private paths/links, symlinks, binaries, weights, compressed-fixture contents, image metadata and broken relative links. No unresolved finding was identified within that scope. A pattern/content audit is not an exhaustive secrecy or legal guarantee.

The upstream MIT notice matches the pinned local source checkout. MIT is prepared for original contributions. DejaVu glyph outlines retain their own complete font notice; model terms remain separate and weights are excluded. See [the license inventory](docs/LICENSE_AUDIT.md). AI-assisted development and historical review scope are disclosed.

The source manifest records every released payload's SHA-256, size and executable mode, retaining the earlier sanitized-fixture provenance. Historical archives and the original C4 ZIP remain unchanged. Full validation logs, source diff and archive-integrity receipts are retained outside the public repository.

## Remaining publication blockers

1. Owner confirmation of licensing authority and the prepared original-contribution MIT grant.
2. Owner confirmation of the GitHub destination, visibility and final content.
3. Explicit publication approval.

No further GPU run is necessary to present the completed measurement/reliability contribution. The original inference-optimization objective remains unfinished and must not be presented as completed. Publish only this curated repository after approval, never the private research workspace or its history.
