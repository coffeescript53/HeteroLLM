# Final portfolio delivery

Prepared 4 November 2026 from the uploaded licensed source release. The original licensed source ZIP has SHA-256 `3ccfbc2c698ca343f512bb7ed61ccef5d3fcc7f746815a2bf50983f017f0cbb3`.

## Deliverables

- Complete project source, versioned C++ instrumentation, tests, preserved measurements, charts, and licenses.
- A rewritten README with a clear contribution map and code tour.
- Final resume wording, shorter alternate, evidence map, and interview opening in `docs/RESUME.md`.
- A readable case study in `docs/CASE_STUDY.md` and responsive interactive page in `site/index.html`.
- A standard-library offline demo that recomputes reported results from bundled integer timings.
- Static portfolio builder with a complete downloadable source archive and evidence links.
- CPU CI, a manual public-only GitHub Pages workflow, and a private-repository uploader.

Production Python modules, instrumentation, test assertions, fixture contents, original measurements, and charts are unchanged. The host-test entry point gained a clear PSI prerequisite check; a separate analysis-only entry point was added. New presentation/build/upload helpers do not perform inference.

## Verification

The original workstation audit recorded 205 passing tests and one expected private-archive skip. Current portable analysis checks passed (32 passes and one expected skip), as did all seven standalone C++ recorder checks. The full selected host suite is not an all-pass result in this cloud container because mandatory `/proc/pressure/memory` telemetry is absent. Test failures were preserved and the supervisor was not weakened. See `docs/VERIFICATION.md` for exact counts and scope.

The offline demo recomputes the published C1 means/sample SDs and C2 paired overhead summaries. The page's chart controls, source tabs, and numerical table were checked in a browser. The source archive includes the real implementation; model weights, compiled upstream runtimes, and private raw archives remain separate.

## Publication state

Public release is deferred. A private portfolio preview is prepared separately. No authenticated GitHub account is available in this session, so the GitHub upload and account-hosted CI/Pages execution remain pending that connection. `docs/PUBLISH.md` and the included uploader make that final account step concrete.

Research status remains current through November 3: baseline and measurement engineering complete within scope; later placement optimization unvalidated. No novel inference speedup or new GPU measurement is claimed.

## Integrity and provenance

`PUBLIC_SOURCE_MANIFEST.json` records the revised source inventory and hashes. The original licensed manifest and delivery note are retained under `docs/release_history/`. MIT and the upstream/font notices remain included. CPU test counts are software checks; they do not diagnose the workstation's GPU health.
