# Experimental dense BF16 Qwen3.5 scoring on CUDA

This proposal adds dense BF16 loading and selected-token scoring to the existing Qwen3.5 CUDA family.
The intended initial geometry is Qwen3.5 27B. It is an engineering candidate, not a qualified checkpoint or release.
The arithmetic proposal and acceptance gates are tracked in [issue #1136](https://github.com/ashhart/TensorFold/issues/1136).
No public checkpoint revision has been qualified by this contribution, and no checkpoint performance claim is made.

## Scope and arithmetic

The loader accepts an unquantized dense configuration declaring `bfloat16`. It rejects unquantized MoE and other
activation dtypes. It reads HF `model.language_model.*` tensors, a tied embedding or separate `lm_head.weight`, and
`[channels, 1, taps]` convolution weights. Norm weights retain the checkpoint's offset representation: the CUDA
RMS and head-norm kernels add one in FP32 rather than rounding the transformed multiplier back to BF16 at load.
Gated DeltaNet's post-normalization weights retain their existing interpretation.

Dense projections use cuBLASLt with FP32 accumulation and BF16 output. Selected-token scoring gathers the final
normalized row of each independent segment, projects those rows to FP32 vocabulary logits, and reduces the entire
vocabulary to log-sum-exp and a deterministic argmax. Equal finite maxima select the lower token ID. Up to 16
selected logits per segment are returned; the normalizer is not restricted to those selected tokens.
The new reduction needs the kernel-contract registration, non-finite policy and device gates before acceptance.
Its finite-input reference utility below does not replace those gates or projection-class FP64 references.

## Development runner

Build with exactly the repository's `.zig-version` toolchain and a supported CUDA compiler:

```sh
zig build qwen35-score -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120
zig build test-qwen35-bf16-host
zig-out/bin/tf-qwen35-score /models/qwen35-bf16 0
```

A complete, matching catalog of prebuilt fatbins can instead be supplied with `-Dfatbins=/opt/tensorfold/fatbins`.
Rebuild `qwen35.fatbin` from this source; the previous release image lacks the scoring symbol.
These paths are illustrative. Keep weights outside the checkout and observe the existing GPU admission guard.
The executable is a Linux JSONL development tool; it is not the native HTTP server or another serving scheduler.

Each newline-terminated input contains one to four independent, already-tokenized requests:

```json
{"requests":[{"input_ids":[1,2,3],"labels":[4,5]}]}
```

The IDs above illustrate the wire schema, not a model prompt fixture. After a readiness line, each batch returns
`results` containing the greedy `token`, selected full-vocabulary-normalized `logprobs` in label order and
`cached_tokens`, plus batch `engine_ms`. This time includes prefix work and device synchronization, but excludes
JSON parsing, output serialization, tokenization, network transport and external queueing.

The runner allocates four 65,536-token work caches and at most eight 64-token prefix caches. Prompts are limited
to 65,530 tokens; that admission limit is not a long-context qualification. Exact first-64-token keys restore
recurrent state and KV data. Remaining prompt tokens and the vocabulary head execute on every operation.
Fresh prefix construction reports zero cached tokens; subsequent reuse reports 64. Prefix entries use FIFO eviction.
Inputs of at most 64 tokens execute afresh. Batches use at most 128 rows per forward.
Invalid input terminates the runner; cancellation, reset recovery, sampling, streaming and vision are unqualified.
The runner owns one device and does not implement tensor parallelism or a multi-device dispatcher.

## Checks and receipt

Port base: `d1018b318fb51981c4a18f12e40269a4d3ee9713`, upstream `main`.
The following Linux x86_64 checks used Zig 0.17.0 and nvcc 13.3.73 for the changed CUDA image. They do not load a checkpoint:

```sh
zig build test test-golden test-cuda-host test-kernel-contract -j4 --summary all
zig build qwen35-score test-qwen35-bf16-host -Doptimize=ReleaseFast -j4 --summary all
python3 tools/zig/lean_check.py
```

- Host/golden/limit suites: 227/227 build steps succeeded; 3,024 tests passed, 14 skipped, zero failed.
- Server corpus: 306/319 equal with 13 recorded known differences; separate corpus 26/26 equal;
  zero unexpected differences.
- A baseline-only BF16 configuration admission probe failed (0/1 passed); the candidate admits that same fixture.
- Final incremental recheck, including targeted tests: 231/231 steps succeeded; 219 tests executed and passed,
  three skipped, with other unchanged checks cached. Zero failures.
- Targeted admission, graph identity and label-range regressions: 3/3 passed, zero failed or skipped.
- Native scorer compiled both host-only and with real embedded CUDA images. The changed Qwen image was rebuilt
  with `nvcc -fatbin -arch=sm_120 -O3 -std=c++17`; other catalog images were retained assets, not rebuilt here.
  Compilation is not a device execution result. No generated binaries or kernel assets are contributed.
- Lean checker: 444 findings on the base and candidate, zero new findings; the absolute checker exit remains nonzero.
- Python reference utility: syntax passed; execution skipped without PyTorch. GPU execution is not claimed.

The finite-input diagnostic below uses synthetic logits only, covers vocabulary tails and ties, and reports each
row's normalizer error against FP64. It requires PyTorch and a free CUDA device; absent dependencies skip cleanly:

```sh
python3 tools/zig/check_qwen35_scores.py /opt/tensorfold/fatbins/qwen35.fatbin
```

An earlier prototype had an unresolved external-reference probability tolerance failure. Agreement on greedy
choices does not establish probability or logit equivalence. Those results are not promoted to a public checkpoint
receipt, and the current port has no claim of checkpoint-level numerical parity.
Before acceptance, use a pinned public checkpoint and public token fixtures to run teacher-forced fidelity,
per-projection FP64 error, prompt/decode seams, resumed versus fresh, batched versus solo and lifecycle checks.
Register the new kernel and post the device contract SUMMARY lines. Run native server decode and cold-prefill
receipts at the required contexts, with cache reuse disclosed separately. Metal, other CUDA SMs, multi-device
execution and checkpoint/device tests remain not run for this port.
