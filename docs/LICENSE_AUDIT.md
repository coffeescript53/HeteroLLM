# Final license audit — 4 November 2026

Scope: the curated public C4-derived source export, with research status through November 3. This is a source/distribution audit by Codex, not legal ownership clearance or an independent scientific review. No original research artifact, runtime, model, algorithm, safety limit or measurement was changed.

## Recommendation

Use **MIT for original HeteroLLM contributions**. It permits reuse, modification and commercial redistribution while requiring the copyright/permission notice. It matches the permissive license of the pinned llama.cpp code. [MIT text](https://opensource.org/license/mit).

Apache-2.0 is also viable for original contributions and adds an express contributor patent grant and patent-litigation termination provision, with more detailed redistribution/change-notice requirements. MIT has no comparable express patent clause. For this small source portfolio, MIT is the simpler fit; it grants no rights the contributor does not own. Neither choice removes third-party obligations. [Apache-2.0 sections 3–4](https://www.apache.org/licenses/LICENSE-2.0).

The prepared root [LICENSE](../LICENSE) identifies “HeteroLLM contributors” without publishing personal information. Confirm that attribution and authority to license original contributions before publication; employment agreements and undocumented third-party provenance cannot be established by a filesystem audit. AI-assisted development is disclosed in [NOTICE.md](../NOTICE.md); no claim is made that every generated fragment independently qualifies for copyright protection.

## Distributed material and obligations

| Material | Classification | Terms and required handling |
|---|---|---|
| `heterollm/*.py`, original tests, scripts, configs, analysis and documentation | Original project supervision, measurement, validation and explanatory work | Root MIT for original portions; retain its notice. No inference-kernel ownership claim. |
| `instrumentation/ggml-cpu-trace.patch`, `instrumentation/v2/ggml-cpu-trace.patch`, `instrumentation/v3/phase-timing.patch` | Mixed original additions and upstream context/modified code | Retain **both** MIT notices. Upstream copyright (c) 2023–2026 The ggml authors remains in `docs/UPSTREAM_LICENSE.txt`. |
| `instrumentation/**/build_patch.py` | Project generators containing upstream source snippets | Same dual attribution boundary as patches. Do not treat embedded upstream code as newly authored. |
| Instrumentation recorder/timing headers, calibration harness and fake-backend tests | Original integration/measurement additions using upstream APIs | Root MIT for additions; keep ggml attribution for any upstream declarations/context. CUDA/HIP/llama headers and native libraries are externally obtained. |
| Sanitized loader/test fixtures and deterministic token-ID fixture | Project-selected test data, with upstream-emitted strings/API formats | Preserve upstream attribution for upstream text. Token IDs are a fixed numeric input, not a tokenizer/vocabulary or model-weight distribution. |
| Compact timing/summary JSON, chart design, architecture Mermaid/SVG and narrative | Project-generated observations and original presentation | MIT for copyrightable original presentation/selection; no exclusive rights claimed over factual measurements. These approved extracts are not private raw experiment archives. |
| `evidence/portfolio/{c1-baselines,c2-overhead,c3-attribution}.svg` | Original charts **plus embedded DejaVu glyph paths** | Glyphs retain separate Bitstream Vera/Arev terms, not MIT. Keep `docs/licenses/DejaVu-LICENSE.txt`; DejaVu changes are public domain. No font binary is bundled. |
| `docs/UPSTREAM_LICENSE.txt`, `docs/licenses/DejaVu-LICENSE.txt` | Third-party license notices | Preserve verbatim; root MIT does not replace them. |

The ggml notice was byte-compared with the pinned local checkout `5e03bdd8700948b9c41c54dd1b00f28a2aebc03f`. Existing attribution was retained. The complete DejaVu notice is copied from the [Matplotlib 3.10.7 font distribution](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.7/lib/matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU), matching the recorded plotting version. Its SHA-256 is `d75938dec098f06f0ac3c00853065d94f020be1c3c62ef1dc2975ba15b4d9b0e`. See also the [DejaVu license](https://dejavu-fonts.github.io/License.html). No chart was regenerated.

## External dependencies — not redistributed here

| Dependency | Use | Licensing boundary |
|---|---|---|
| Python standard library | Core CLI, tests, analysis and supervision | Python/PSF terms; no Python interpreter/library vendored. |
| setuptools >=77 | Python packaging backend | MIT; installed by the user. Package metadata includes the applicable notice files. PEP 639 metadata requires setuptools >=77; no installed dependency was upgraded. The Python distribution is MIT; the repository-wide third-party exceptions remain explicit in NOTICE.md. |
| Matplotlib | Optional chart rendering only | PSF-based Matplotlib license and dependency-specific licenses. Its package and transitive dependencies are not shipped; embedded DejaVu outlines are addressed separately above. |
| llama.cpp/GGML pinned source | Optional native loading, kernels, scheduler and benchmark runtime | Upstream MIT plus any third-party components in a separately acquired upstream checkout. This export distributes patches, not the whole checkout or compiled runtime. |
| CUDA Toolkit / NVIDIA driver; ROCm/HIP / AMD driver | Optional GPU build/execution | Component/vendor terms apply separately. No SDK headers, driver, runtime library or compiler binary is bundled. A future binary release requires a separate dependency/redistribution audit. |
| C++ compiler, CMake, shell and Linux interfaces | CPU recorder tests / optional native build | External tools under their own licenses; invoking them does not relicense this source. |
| Meta Llama 3.3 / bartowski Q4_K_M quantization | Original research workload, separately acquired | Llama 3.3 Community License and use policy, not MIT. Weights, tokenizer vocabulary and Meta implementation code are excluded. |

The core package declares no third-party Python runtime dependencies. Import inspection found Matplotlib only in the optional plotter; the other imports are Python standard-library or project modules. This is not an SBOM for a separately installed native GPU environment.

Llama use remains subject to Meta's agreement. A later distribution containing Llama materials or derivatives must include the agreement and attribution and meet the “Built with Llama” and use-policy obligations. An MIT code license does not authorize model redistribution or hosted deployment. This source-only portfolio does not bundle those materials. [Meta Llama 3.3 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE).

## Verification and remaining boundary

Both documented CPU entrypoints were rerun: **205 passed, one expected private-archive skip**. The selected suite uses Linux procfs/PSI and owned fake CPU subprocesses; the optional recorder tests compile standalone C++17 fixtures. No GPU/native inference runtime was loaded. An offline Python wheel build checked the MIT license expression, all four notice files byte-for-byte and the Python-package-only inventory; no package was installed or published. Current setuptools emitted no deprecation warnings.

The final inventory audit examines ordinary files, decoded gzip fixtures, image metadata, links, executable modes and the staged file list. No private home paths, private Drive URLs, recognized credential material, weights, compiled binaries, private execution contracts, bulk numerical captures, raw experiment directories or private Git history are included. Approved compact benchmark extracts and sanitized parser fixtures are intentionally retained. Pattern checks cannot prove absence of every possible secret or undocumented copyright claim.

No unresolved license-text/attribution defect was identified within this source-export scope after adding the font notice. Remaining owner decisions: confirm rights and the prepared MIT grant, select repository/account, choose a public commit identity and explicitly authorize publication. GitHub CLI is not installed in the audited environment; the publication instructions use Git plus creation of an empty repository in GitHub's UI. No remote, push, publication or visibility change occurred. See [PUBLISH.md](PUBLISH.md).
