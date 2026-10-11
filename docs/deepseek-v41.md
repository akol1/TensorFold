# DeepSeek V4.1 Flash: experimental resident Metal draft

**Not release-qualified. Keep the proposal in draft until the numerical and runtime gates pass.**
[Proposal #685](https://github.com/ashhart/TensorFold/issues/685) covers this implementation.
The backend executes text inference; it is not a placeholder. Its independent whole-model
comparison still fails. Greedy-token agreement does not establish numerical correctness.

## Qualification boundary

The following receipts are historical, from main
`bec00ae6961a43a9ab73c4fa9338c167b3f82851` plus the local DS4.1 candidate changes.
This draft integrates that work onto main `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`.
They are not new verification of the integrated draft. Current-branch checks are reported in the PR.

| Retained measurement | Result | Qualification |
| --- | --- | --- |
| Eight-token whole-model NRMS | 0.040960584768951355 (4.0960584769%) | FAIL; unchanged limit 0.01 (1%) |
| Maximum absolute logit error | 1.5123639106750488 | Not a substitute for NRMS |
| Greedy disagreements | 0 of 8 | Does not clear numerical gate |
| Teacher replay / independent CPU oracle | 5.350994958 s / 346.856063250 s | Not served-decode throughput or speedup |
| Operator checks | 61 Metal + 11 independent reference | Limited operator evidence |
| 257-token chunk/history replay | Identical FP32 logits and greedy tokens | Chunks 1, 2, 3, 7, 31, 127, 128, 129, 255, 256 |
| Same history replay residency | 0 read bytes; 0 page-ins | Passing measured execution scope only |
| Separate full trace residency | 20,480 read bytes; 0 page-ins | FAIL; not a zero-I/O receipt |

`TeacherReferenceMismatch` remains a real failure, not a warning to suppress.
The [historical evidence record](deepseek-v41-evidence.md) preserves intermediate results,
rejected experiments, CUDA counterexamples and earlier speed receipts.

### Unresolved numerical causes

- A retained HC exponent input `1.4393715858459473` gives Metal FP32 `0x4086fa39`
  versus source CUDA and ideal-rounded `0x4086fa38`. Post-coefficient 3 changes a BF16
  expansion cell from source 14831 to native 14832. Replacing only the actual source
  coefficient fixes that retained cell; qualification never injects source coefficients.
- Six Metal exponent variants were exercised on 16 retained arguments and 65,537 grid
  points from -32 through 32. None fixed the controlling cell. Single/split-log2 `exp2`
  increased worst differences to 29/22 ULP versus production's 2 ULP; all were rejected.
- Independent wide reductions and ideal FP32 unary rounding are not a CUDA hardware
  emulator. Actual source-CUDA execution also agrees with native on some cells where
  the independent oracle differs. Lower NRMS alone does not justify changing arithmetic.
- The canonical 64-head CUDA sparse kernel needs 141,312 bytes of dynamic shared memory;
  the available GB10 permits 101,376. Compilation succeeded but launch did not.
  Executed 16-head checks are operator evidence, not full-head or whole-model qualification.

## Implementation and limits

`zig/src/families/deepseek_v41/` owns configuration, resident weights, workspace,
hyperconnections, Engram, sparse/compressed attention, routed/shared experts, RoPE and
history state. `zig/kernels/metal/deepseek_v41/` implements device operators.
The native host uses the existing serialized worker and HTTP transports. Independent
qualification lives in `zig/tests/deepseek_v41_reference/`; it does not import native
statistics or replace its wide reduction with the backend's reduction tree.
Cancellation is observed between bounded prompt/decode chunks; it does not interrupt
an in-flight Metal command buffer. The serialized host owns model/history state until
that command completes before reset, reuse or teardown.

Shared EXL3 APIs check descriptor arithmetic, aligned view extents and writable-view
nonaliasing. Callers must supply actual input/output capacities for GPU ID maps,
guarantee IDs are in range and scatter output IDs are unique, register indirect weights
resident, and keep indirect descriptor-referenced weights disjoint from writable buffers.
Raw family encoders are trusted fixed-shape internals, not unchecked external APIs.
Generic counters classify `gpu_profile.Class`; the family profiler owns DS stage metadata
and totals without allocation or copied spans. Counter cleanup is fallible while encoding
or execution is in flight. There are no per-span CPU waits or fabricated timing fallbacks.
Shared EXL3 projections and hardware counters are separate from family arithmetic.

The complete EXL3 checkpoint and original Engram tables are owned resident Metal buffers.
There is no weight mmap or per-request SSD table streaming. Startup reads are expected;
steady-state resident claims require measured process read/page-in deltas. Admission
leaves an 8 GiB reserve below Metal's recommended working set and rejects incomplete fit.

Only one active sequence is supported; concurrent clients queue. Vision execution,
draft heads, thinking budgets, Sliding/Living Weights, learned states and retained prompt
caches are unsupported. Request admission must reject unsupported combinations.
`--context` includes prompt plus reply, from 1 through 1,048,576; prompt chunks are at most
256 rows. The packed cache uses 890 bytes per token plus bounded window/compression state.
Allocated or advertised context is not evidence that every position was exercised.
A historical prefill passed 128k tokens, then was intentionally cancelled before generation;
full million-token execution remains unqualified.

On the measured M3 Ultra with 512 GiB RAM, the weight payload was 413,338,642,120 bytes
(384.95 GiB). Full-window cache, scratch and resident RoPE factors brought admission to
415,199,770,312 bytes (386.69 GiB). RoPE factors account for 512 MiB at maximum context;
they are not weight bytes and incur no per-request CPU trigonometry or factor transfer.
Use `/memory`, process counters and `vmmap` together; aliased virtual mappings are not
additive physical memory, and a read delta alone does not prove weight streaming.

## Pinned inputs and build

Keep snapshots outside the checkout. This PR contains no model shards or binary traces.

- EXL3: [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw/tree/64ba41b6c916a587db06eae2e19b7845f7be6e6b),
  revision `64ba41b6c916a587db06eae2e19b7845f7be6e6b`, complete snapshot.
- Source: [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/2cba9e42aa026125f3ed06c6d98c1db82f7ca027),
  revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`. Supply
  `model-00047-of-00048.safetensors`, `model-00048-of-00048.safetensors`, `config.json`,
  `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`, `LICENSE`, and `inference/config.json` for the offline map builder.
  Only `.engram.embed.weight` and `.engram.embed.scale` come from these shards;
  projections come from EXL3.

Use Zig 0.17.0 and Xcode's macOS SDK/Metal toolchain:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  zig build native tf-dsv41-run -Dcpu=apple_m1 -Doptimize=fast
python3 -m venv .venv-weights
.venv-weights/bin/pip install numpy==2.5.3 tokenizers==0.23.3
.venv-weights/bin/python tools/zig/dsv41_token_map.py \
  --tokenizer /models/dsv41-exl3/tokenizer.json \
  --config /models/dsv41-original/inference/config.json \
  --output /models/dsv41-exl3/engram-token-map.bin
```

The offline map builder preserves official normalization, PCG64 multipliers and prime
bucket layout. Native inference requires no Python. Use the pinned original template
explicitly; do not silently substitute the EXL3 snapshot's older template.
The map builder requires the original inference config's field names, not the converted
model config schema. Both pinned configs use 16,000,000 Engram buckets and 129,280 token IDs.

## Reproduce diagnostics

Run one full-model process at a time. `MODEL` means `/models/dsv41-exl3` and `TABLES`
means `/models/dsv41-original`; these paths are examples, not required directory names.
Use fresh output directories and preserve exit status plus JSON receipts.

Exact retained teacher input, saved as `teacher.json`:

```json
[0, 671, 6102, 294, 8760, 344, 11111, 16]
```

The retained 257-token seam input is exactly `(teacher_ids * 33)[:257]`:

```sh
python3 -c 'import json; ids=json.load(open("teacher.json")); json.dump((ids*33)[:257],open("seams-257.json","w"))'
zig-out/bin/tf-dsv41-run plan /models/dsv41-exl3 /models/dsv41-original
zig-out/bin/tf-dsv41-run teacher /models/dsv41-exl3 /models/dsv41-original teacher.json /tmp/dsv41-teacher 1
zig-out/bin/tf-dsv41-run oracle /models/dsv41-exl3 /models/dsv41-original teacher.json /tmp/dsv41-oracle
zig-out/bin/tf-dsv41-run seams /models/dsv41-exl3 /models/dsv41-original seams-257.json /tmp/dsv41-seams
zig build test-deepseek-v41 test-deepseek-v41-reference -Dcpu=apple_m1 -Doptimize=fast
```

A failed oracle exits with `TeacherReferenceMismatch`; do not rerun merely to relabel it.
The oracle's independent-chain output is qualification. Native-input or native-coefficient
reuse is an explicitly diagnostic counterfactual and cannot replace that output.
Teacher/seams accept at most 4,096 IDs; oracle/trace at most 16; profile at most 256.
IDs are nonempty JSON arrays of integer vocabulary IDs below 129,280.

The runner also provides `generate`, `profile`, `trace`, `operators`, `attention-sameinput`,
`attention-history-sameinput`, `hc-sameinput` and `moe-sameinput`; invoke it without
arguments for syntax. History replay needs `trace-{position}-gpu.bin` for every position
through the requested token, not an invented cache or a single late snapshot.
`moe-sameinput` defaults to layer 21 and cannot infer token identity from a headerless snapshot.
Counter capture may rotate encoders and change submission overhead; profile timing is
not an uninstrumented throughput measurement. Receipts distinguish `model_execution`
from `execution_and_diagnostic_io`; never compare scope differences as weight-streaming proof.

## Exercise actual serving

Start with a bounded context; the complete resident weight payload still requires the
large-memory machine. The example is intentionally loopback-only and thinking off:

```sh
zig-out/native/bin/tensorfold-native serve /models/dsv41-exl3 \
  --engram-table-dir /models/dsv41-original \
  --chat-template /models/dsv41-original/chat_template.jinja \
  --context 4096 --parallel 1 --no-drafts --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8080 --name DeepSeek-V4.1-Flash \
  --keep-warm 0 --no-update-check
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Respond with only the word READY."}],"temperature":0,"max_tokens":24,"chat_template_kwargs":{"enable_thinking":false}}'
```

For throughput receipts, preserve public prompt/token IDs, uncached counts, sampling
settings, output count, first/last token times, exact executable commit and hardware.
Decode rate excludes the first token. Do not turn single-run smoke receipts into a
statistical comparison or a whole-model numerical qualification claim.

## Integrated draft smoke receipt

These checks exercised the integrated implementation, not the older numerical campaign.
Environment: Apple M3 Ultra, 512 GiB RAM, macOS 26.6.2, Zig 0.17.0 and Xcode Metal;
the PR records the exact published commit. Checkpoints/revisions are the pinned inputs above.
The known failed whole-model comparison was not rerun merely to reconfirm failure.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
TENSORFOLD_V41_TOKENIZER_DIR=/models/dsv41-original \
  zig build native tf-dsv41-run tf-exl3-bench test-deepseek-v41 \
    test-deepseek-v41-reference test-server test-golden test \
    -Dcpu=apple_m1 -Doptimize=fast -j2 --summary all
zig-out/bin/tf-exl3-bench --shape 4 --warmup 1 --repeats 1 --prefill 2
```

- Final build: 158/158 steps succeeded; 178 newly executed tests passed, with remaining
  successful test steps reused from cache. Dedicated DS checks passed 75 device/host
  cases and 13 independent reference cases during integration; these are not 88
  whole-model comparisons. The server consumer suite passed 155/155.
- Frozen HTTP goldens: 309/319 exact matches, 10 explicitly known differences,
  zero unexpected differences. No golden answers were rewritten.
- Actual EXL3 CLI: all 196,608 codebook states matched FP16 bits; the selected
  `K=4096,N=1024` one-row, routed-six and two-row prefill paths had zero comparator
  bit mismatches and passed the sampled independent FP64 oracle gate. One repeat
  is smoke evidence, not a statistically supported speed claim.
- Actual offline map packing produced 517,680 bytes, raw/compressed vocabularies
  129,280/99,092 and compressed pad 2. Metadata/lookup matched the retained map;
  the only 32 differing bytes were the correctly refreshed input-config SHA-256.
- Actual `plan` admitted 41 shards, 192,456 tensors and 413,338,642,120 weight bytes.
  Actual serving at context 4,096 admitted 413,735,409,352 device bytes and printed
  `EXPERIMENTAL / UNQUALIFIED`. Startup took 113.9 seconds; this is not decode latency.
- Greedy, thinking-off whole and SSE requests for “Respond with only the word READY.”
  returned `READY`, with 12 prompt tokens, 3 completion tokens and zero cached tokens.
  An intervening “What is 2 + 3? Respond with only the answer.” returned `5`;
  repeating the first request preserved content/counts and its logged token-hash prefix.
- Whole and SSE requests for “Call lookup with q set to Paris. Do not answer directly.”
  with one `lookup(q:string)` tool and `parallel_tool_calls:false` returned exactly one
  call, `{"q":"Paris"}`, index 0 in SSE, and `finish_reason:"tool_calls"`.
  Both used 275 prompt tokens and 38 completion tokens, zero cached tokens;
  this actual prompt crossed the 256-row prefill boundary.
- Actual requests rejected positive thinking budgets while thinking was off, explicit
  drafts, image input, and a 4,096-token prompt leaving no reply room with HTTP 400.
  Closing a counting stream after its first content delta logged `reason=client-left`;
  the next request returned `READY` normally. SIGTERM shutdown exited 0.
- Process `proc_pid_rusage` deltas were zero read bytes and zero page-ins across the
  three text/reset requests, across disconnect/recovery, and across the two tool requests.
  These bounded request intervals do not reclassify the historical failed full trace,
  qualify long-context execution, or clear the whole-model numerical gate.
- Source-rule scan: 1,241 files, 532 inherited findings versus main's recorded
  1,169 files/534 findings; zero added findings. The checker remains nonzero,
  not “lint clean.” Outgoing text contained no private home paths or credential markers;
  weights, binary traces, build outputs and local receipts were excluded.

## Publication and ready gate

Shared contributions overlap #648, #649, #650, #680, #681 and #684. This draft is based
on main, not on unmerged PR heads; overlap must be reconciled before merge.
The qualified-model table remains unchanged. The PR stays draft while any gate below fails:

- Unchanged independent whole-model NRMS gate passes on the retained teacher IDs,
  without coefficient injection, oracle/backend conflation or omitted history.
- Remaining source-CUDA unary discrepancy is resolved with consumer-visible evidence;
  unsupported full-head/source-backend checks are explicitly completed or scoped by reviewers.
- Affected operator, reference, server and golden checks pass on the integrated commit.
- Actual serving, reset/cancellation, admission and relevant chunk/history paths are exercised.
- Zero-I/O receipts identify their scope; long-context claims match exercised positions.
- Outgoing files/history contain no credentials, private hostnames, local paths, weights,
  binary traces or generated build artifacts. Public provenance and license notices remain.

No automated action in this work should mark the PR ready, merge it, or advertise the
family as release-qualified. Further DS4.1 numerical investigation continues locally.
