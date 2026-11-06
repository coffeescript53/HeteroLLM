# Five-minute project walkthrough

1. Read the README result table and C1 chart. Explain why CPU-offload baselines and all-GPU mixed placement are different deployment configurations, not an original kernel speedup.
2. Follow the architecture diagram from CUDA blocks through host staging to ROCm blocks. Explain why equal split is not a literal 40/40 transformer split.
3. Show the C2 correctness and overhead summaries. Distinguish fixed-input bitwise agreement from general model correctness and six process pairs from correlated internal repetitions.
4. Show the C3 chart. The overlapping CPU/device scopes do not add to wall time. A longer ROCm region does not establish a removable operator bottleneck.
5. Close with one debugging example and the [current research status](STATUS_REPORT.md): C3.2 reached a profiler NO-GO; C5's frozen numerical comparison failed; the November 3 exploratory retry stopped when the AMD junction reached 86°C against the 85°C guard. P did not launch and no valid E/P quality comparison resulted. Explain how durable records and cleanup preserve a useful failure without turning it into a performance claim. A validated source-level optimization remains unfinished.

Use the [interview guide](INTERVIEW_GUIDE.md) for detailed questions. Everything above can be presented offline from the public charts and source. No model loading, GPU discovery or benchmark is necessary.

For a live coding demonstration, run `python3 scripts/demo.py` to replay the published arithmetic and `bash scripts/test-offline.sh` for portable analysis fixtures. The broader `bash scripts/test-cpu.sh` needs Linux memory-pressure telemetry. The synthetic snapshot preview in the [reproduction guide](REPRODUCE.md) is another offline option. These exercise parsing, safety failures, cleanup, statistical arithmetic and offline trace validation with CPU fixtures. Their artificial timings are never benchmark evidence.
