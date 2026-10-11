# DeepSeek V4.1 historical numerical and runtime evidence

These are retained investigation receipts, not verification of the current-main draft.
The measured source was `bec00ae6961a43a9ab73c4fa9338c167b3f82851` plus the local
DS4.1 candidate changes; it was not a clean committed release. The draft integrates
that work onto main `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`.
Do not cite intermediate test counts or lower historical errors as the current gate.
The latest retained gate is **4.0960584769% NRMS against an unchanged 1% limit: FAIL**.
The [draft guide](deepseek-v41.md) states current boundaries and reproduction commands.

The following chronology preserves rejected experiments and negative results. It is
not a speedup claim, a source-CUDA parity claim, or a passing residency receipt.
Large local traces, checkpoint weights, and private filesystem paths are not included.

## Retained investigation

The eight-token teacher comparison initially measured 7.6812% normalized RMS error. Intermediate reference
corrections measured 2.8089% and 2.7056%, but those are not the current qualification result. Preserving the
source compressor's FP32 softmax-before-weighted-sum boundary measured 5.4496%; aligning native attention's
tile-denominator and sink arithmetic with the pinned source measured 6.1794%. Correcting native and reference
SiLU division ordering then measured 3.4807%. Revisiting compensated native EXL3 accumulation after
same-input attention isolation eliminated all first-layer attention-branch differences and measured 3.2922%
whole-model NRMS. A scalar compensated-dot candidate then matched all ten projection checks' FP32 outputs
from the independent FP64-dot oracle, but measured 3.6140% whole-model NRMS. Correcting the reference's
FP32 accumulator-before-mean-division boundary measured 3.8218%. Compensating native FP32-weight dense
products and reductions then removed the retained layer-4 hyperconnection projection error and its
downstream layer-5 normalized-input BF16 mismatch, measuring 3.5010% whole-model NRMS. Running the
pinned TileLang 0.1.8 HC kernel on CUDA and inspecting its generated PTX then established fused affine
multiply-adds and FP32 butterfly denominator reductions. Applying those contracts to native HC and
the independent reference measured 3.7623% eight-token whole-model NRMS. Computing the oracle's unary
inverse square root independently in FP64 and rounding its result once to FP32 then measured 4.1783%,
with maximum absolute logit error 1.583322 and zero greedy disagreements. Correcting the reference's
HC collapse/post FP32 accumulation boundary and unary power rounding then measured 4.4685% NRMS,
maximum absolute error 2.321231, and zero greedy disagreements across all eight tokens. Disabling
implicit FMA contraction in native HC collapse/post then removed the isolated layer-11 BF16 expansion
error, but measured 4.7181% whole-model NRMS, maximum absolute error 2.052191, and zero greedy
disagreements. Compensating native sparse QK accumulation and lane reductions then measured
2.9347% NRMS, maximum absolute error 1.153051, and zero greedy disagreements across all eight tokens.
Correcting native six-score routing normalization to its independently established source-CUDA
FP32 reduction then measured 1.2571% NRMS, maximum absolute error 0.488874, and zero greedy
disagreements. Precomputing resident source-style complex RoPE factors then fixes the retained
token-5/layer-0 attention case, but raises whole-model NRMS to 3.9876%, maximum absolute error
1.563581, with zero greedy disagreements. Correcting native HC's statistic to the independently
executed canonical single-row CUDA reduction then measures 3.7143% NRMS, maximum absolute
error 1.373373, and zero greedy disagreements. Correcting the CUDA mean scalar projection
in ordinary RMS, Engram and the independent reference then measures 4.0961% NRMS, maximum
absolute error 1.512364, and zero greedy disagreements. This is the current candidate, not
the lowest historical metric. The unchanged admission gate is 1%; `TeacherReferenceMismatch`
remains an upstream blocker. The ideal unary oracle is not CUDA hardware emulation.
The latest GPU teacher replay took 5.351 seconds; this is not served-decode throughput
or a comparative speedup.
The earlier compensated-accumulation experiment had been reverted after a worse controlled whole-model
result with the then-current reference. A proposed single-round inverse-square-root reference change was
also reverted: PyTorch 2.14.1 CPU and MPS matched FP32 square-root followed by reciprocal on all 10,001
positive probe inputs, not the proposed FP64 inverse-square-root rounded once. That probe does not establish
CUDA intrinsic rounding. These numerical runs are not new API speed measurements.

The mean-boundary probe matches PyTorch 2.14.1 CPU and MPS on the retained layer-4 statistic.
The reference still uses independent wide sums internally, not either backend's FP32 reduction tree.
Actual PyTorch 2.13.0a0 / CUDA 13.3 probing on an NVIDIA GB10 shows that CUDA `rsqrt` differs from
CPU/MPS square-root-then-reciprocal rounding on 3,372 of 10,003 positive inputs, and from a correctly
rounded FP64 inverse on 2,255. Both retained inverse examples agree with the original Metal intrinsic.
Compensating native HC variance measured 3.9120% whole-model NRMS without removing the retained
inverse discrepancy; a CPU/MPS-only native inverse rewrite measured 3.4688%. Both experiments were
reverted: the first did not address the causal discrepancy, and the second is not justified by the
actual CUDA operation, irrespective of its lower metric. CUDA mean also uses a rounded reciprocal
factor rather than CPU/MPS mean division. A complete source-backend reference remains unqualified.
The FP64 projection oracle and admission threshold have not been relaxed. Dense cancellation
regressions fail before compensation and pass after it.
The earlier inverse-reference experiment was based only on CPU/MPS behavior. Actual CUDA evidence now
supports retaining a distinct ideal unary FP32 result instead of adding an intermediate FP32 square-root
rounding to the mathematical oracle. On the retained layer-5 statistic that removes the one-ULP inverse
discrepancy; pre/post/comb coefficient errors remain below 2.8e-9. The 10,003-input CUDA probe above
still rules out claiming bitwise source-CUDA parity for this ideal reference.

The TileLang probe uses independently computed FP64 projection outputs rounded to FP32 as inputs;
it is an isolated source-kernel check, not a whole-model CUDA comparison. Generated PTX confirms
`fma.rn.f32` affine operations and descending XOR row/column reductions. The corrected retained
layer-4 HC comparison has maximum coefficient errors of 4.66e-10 (pre), 3.73e-9 (post), and
1.86e-9 (comb), with identical next collapsed and normalized BF16 rows. Those checks do not clear the
whole-model gate; remaining CPU/reference versus CUDA primitive differences still require resolution.

Executing the pinned `Block.hc_pre` and `Block.hc_post` methods on CUDA also establishes FP32
accumulation of their materialized FP32 products. Two half-ULP terms are lost when accumulated
after 1.0; summing them in FP64 instead changes a cancelling BF16 result from zero to 2^-23.
The reference now retains the source FP32 accumulation boundary. Its test target previously
omitted the newer HC, RoPE and inverse-square-root regressions; enabling them reproduced the
residual failure and a two-ULP frequency error. The reference power operation now computes an
independent ideal unary result rounded to FP32 before reciprocal. All 64 retained CPU
frequency-vector values pass the unchanged test tolerance; this is not source-CUDA bitwise qualification.

The native collapse/post product boundaries also require disabling implicit FMA contraction.
With identical retained layer-11 input, branch and native coefficients, the old expansion rounded
one BF16 cell to 26.625 instead of 26.5. Reversing the cancellation fixture reproduced the contraction
error: BF16 0x34c0 instead of 0x3500. Actual pinned CUDA collapse, residual and branch-product
cancellations all produce 2^-21. The corrected native retained expansion matches all 20,480 BF16
cells; all 56 Metal checks and 10 independent reference checks pass. Explicit HC affine FMAs remain.

The retained layer-21 compressed-attention isolation replays the normalized prefix to populate
the real compressed cache. Queries, window keys and compressed keys match, but uncompensated
native QK accumulation changed one sparse-output BF16 cell from 0.38671875 to 0.388671875.
The pinned CUDA sparse kernel's 16-head specialization agrees with the independent reference.
A separate cancellation fixture loses a 2^-6 term beside cancelling 2^24 terms in native lane
accumulation, changing sink-normalized output from source BF16 0.50390625 to 0.5. Compensated
per-lane accumulation and lane reduction reproduce the source result; the regression fails before
and passes after the correction. All retained layer-21 attention-stage BF16 outputs now match,
and all 57 Metal plus 10 reference checks pass.

These CUDA probes use TileLang 0.1.8 with its required Apache TVM FFI 0.1.6. Re-running the
HC source check with that supported dependency preserves every previously observed coefficient
bit. The full 64-head sparse kernel compiles but cannot launch on the GB10: it requests 141,312
dynamic shared-memory bytes, exceeding the device's 101,376-byte opt-in limit. The executed
16-head specialization is isolated operator evidence, not full-head or whole-model qualification.

The source-CUDA routing probe establishes the contiguous six-score sum as
`((s0+s4)+s2)+((s1+s5)+s3)`: four lane accumulators followed by descending shuffle offsets 2 and 1.
Actual CUDA execution matches this reduction on 8,918 positive, signed and permuted boundary
rows; sequential and wide reductions disagree on 2,815 and 2,468 rows respectively. Two
consumer regression fixtures reconstruct exact scores through above-threshold softplus and
square root, avoiding a unary-rounding confound. Native normalization now follows the source
contract; the independent mathematical oracle retains wide normalization rather than copying
the backend reduction tree.

`tf-dsv41-run moe-sameinput MODEL TABLES TRACE_GPU_BIN OUT_DIR [LAYER]` isolates
routing, six routed experts and the shared expert on a retained normalized input; layer defaults
to 21. It records independent-chain comparisons separately from native-input/native-weight
diagnostic reuse. Raw snapshots have no token header, so this history-free MoE receipt reports
token identity as unknown and preserves the supplied snapshot path rather than claiming token zero.
The corrected retained layer-21 logits, weights, gate/up projections,
activations, down projections and all 5,120 combined-branch BF16 cells match the independent
reference. Replaying the retained old weights reproduces all 11 old branch differences.
All 58 Metal and 10 reference checks pass; those operator checks do not clear the full-model gate.

`tf-dsv41-run attention-history-sameinput MODEL TABLES TRACE_DIR OUT_DIR TOKEN LAYER`
replays retained normalized attention inputs at every position through the selected layer,
advancing real native and independent reference caches together. It exports queries, physical
window rows, compressed keys, selected indices and sinks for separate source-kernel replay;
native-query/cache reuse is explicitly diagnostic and never enters qualification.
At token 2, layer 1, all attention-stage BF16 outputs and all 13 isolated FP32 projections match;
the replayed native branch matches the retained full trace, with zero disk reads and page-ins.
Thus the original layer-1 mismatch at that token depends on earlier differing history inputs,
not a demonstrated operator error on identical history. This does not clear the eight-token gate.

The earlier ideal-unary first-token diagnostic preserved native logits, but its residency check
failed: process disk reads increased by 12,288 bytes with zero page-ins. The first layer-11 isolation
also failed its residency check with 4,096 read bytes. Neither receipt establishes weight streaming.
The corrected-reference first-token trace and corrected native layer-11 isolation both measured
zero process disk-read bytes and zero page-ins; those limited runs do not qualify every workload.
The compressed-attention isolation runs measured 8,192, 4,096 and, after QK compensation,
16,384 process disk-read bytes respectively, each with zero page-ins. Their numerical evidence
does not make them passing zero-I/O qualification receipts or establish weight streaming.
The QK-compensated full eight-token layer trace preserves every native logit bit, but also fails
residency with 16,384 read bytes and zero page-ins. Its first persistent first-token divergence
was layer-21 MoE with identical normalized input; the routing correction removes that isolated
discrepancy. The corrected MoE isolation also fails residency with 4,096 read bytes and zero
page-ins. Neither numerical isolation is a passing zero-I/O receipt.
The routing-corrected eight-token trace also preserves all native logit bits, but fails residency
with 20,480 read bytes and zero page-ins. The first persistent first-token difference is now one
layer-24 HC post-expansion cell; later teacher tokens also diverge in history-dependent attention.
Retained layer-24 HC projection and collapse/post with identical native coefficients match exactly;
the inverse differs by one FP32 ULP from the ideal reference. Actual intermediate-statistic capture
then shows native and pinned Torch CUDA agree: sum 561,016,537,088, mean/argument 27,393,386,
inverse 0.00019106324180029333. The wide oracle rounds its sum to 561,016,602,624 and argument
to 27,393,388; changing that argument alone explains the inverse difference. Metal `rsqrt` agrees
with ideal unary rounding on both captured arguments, so variance compensation or an inverse
rewrite is not a source-backed correction. The isolation measures zero reads and page-ins,
but this retained-case CUDA parity does not clear the whole-model gate.

A separate token-1/layer-0 attention probe exposes a real native HC statistic mismatch: squared
sum 28.332973480224609 versus actual pinned CUDA 28.332975387573242. CUDA matches the wide
oracle on that input. Native now follows CUDA's canonical single-row contiguous FP32 mean:
512 threads, four independent vector accumulators, descending block/warp reductions, then
FP32 mean-factor multiplication. Independent CUDA probes establish the tree on both retained
inputs and ten generated inputs; standalone native execution matches the source single-row
inverse for each input at all 14 tested row counts. The older HC24 source result is preserved.
Changing only the token-1 inverse fixes both retained expansion cells; production-tree replay
matches all 20,480 reference BF16 cells. A permanent regression recomputes RMS, coefficients
and post-expansion from the retained input rather than injecting the oracle inverse.
Native fixes the same per-row tree across chunks: CUDA can choose another reduction grouping
for batched rows, so this is decode-tree parity, not every batched-CUDA rounding behavior.
The integrated build passes all 60 Metal and 10 reference checks. The actual full 257-token
history preserves every FP32 logit bit and greedy token across chunks 1, 2, 3, 7, 31, 127, 128,
129, 255 and 256, with zero execution disk reads and page-ins. The eight-token whole-model
numerical gate nevertheless remains failed at 3.7143% NRMS.

Pinned Torch CUDA exposes a separate mean-projection boundary: with BF16 input `[3, 0, ...]`
of width 5,120, exact squared sum 9 and FP32 gamma bits `0x3c65de29`, canonical
`RMSNorm.forward` returns BF16 1.0 on CUDA but 1.0078125 on CPU. CUDA materializes the
FP32 reciprocal count and multiplies the FP32 accumulator; division crosses this affine
rounding tie. The native ordinary RMS reproduces the CPU result before correction and
the CUDA result afterward. Native Engram uses the same corrected mean projection.
The reference now preserves that scalar boundary around its independent FP64 sums for
ordinary RMS, HC and Engram; it does not reuse native statistics or reduction trees.
All 61 Metal and 11 reference checks pass. The eight actual initial teacher-embedding
RMS cases already matched at every BF16 cell, so this constructed consumer does not
establish the whole-model root cause. The corrected candidate's actual full 257-token
history remains bit-exact across chunks 1, 2, 3, 7, 31, 127, 128, 129, 255 and 256, with
zero execution disk reads and page-ins. The full eight-token gate nevertheless fails
at 4.0961% NRMS. Its separate trace preserves every native logit bit but records 20,480
execution disk-read bytes and zero page-ins, so it is not a zero-I/O receipt.

Later ordinary RMS isolation also rules out a reduction rewrite for token-1/layer-2:
production replay reproduces all 5,120 retained normalized values, and exact pinned
`RMSNorm.forward` CUDA matches native on the identical real collapsed input and BF16
norm weights. CUDA collapse with fixed native pre-coefficients is bit-exact too.
The delayed discrepancy instead comes from layer-1 FFN HC pre-coefficients: replacing
only those with independently computed wide coefficients changes collapsed cell 1,580
and exactly reproduces the retained reference normalization. Native HC projection and
inverse match the wide calculation; canonical HC coefficient execution must establish
whether the remaining unary difference is a native error.

The earliest token-0/layer-16 MoE branch discrepancy is source-conformant, not a justified
routing rewrite. Gate/up projections are bit-exact; wide normalization changes two BF16
expert activations and then one combined-branch cell. Actual pinned Torch CPU and CUDA
Gate/Expert tails on the retained projections match all 16,128 native activation cells.
The diagnostic using native weights matches every expert stage and branch, but never
replaces qualification. This isolation records 12,288 read bytes and zero page-ins.

Canonical TileLang execution separates the token-1/layer-0 FFN's three differing
expansion cells. Native's inverse matches actual pinned CUDA, not the wide statistic;
CUDA therefore agrees with native at cells 9,998 and 16,906. The third cell, 18,987,
remains a genuine unary-consumer mismatch: controlling exponent input 1.4393715858459473
produces Metal FP32 bits `0x4086fa39`, versus source CUDA and ideal-rounded `0x4086fa38`.
That difference changes post-coefficient 3 and BF16 expansion 14,832 versus source 14,831.
Substituting only the actual source post-coefficient fixes that cell; substituting source
comb coefficients does not. No source coefficient is injected into qualification.
The delayed layer-2 normalization discrepancy also uses a pre-coefficient where actual
TileLang agrees with native, not the wide reference.

Six actual Metal exponent variants were checked on the 16 retained HC arguments and
65,537 FP32 grid points from -32 through 32, against pinned Torch CUDA and ideal rounding.
Explicit `precise::exp` does not change production output; single- and split-log2 `exp2`
substitutions do not fix the controlling cell and increase source disagreements, reaching
29 and 22 ULP respectively versus production's 2-ULP maximum on this campaign.
All substitutes are rejected. A portable correction to this HC unary boundary and the
whole-model discrepancy remain unresolved; the unchanged numerical gate still blocks
ready-for-review or merge of the family implementation.

The retained token-5/layer-0 query differs in one BF16 cell before this RoPE correction: native
0.54296875 versus source/reference 0.546875. Exact pinned Torch CPU and CUDA operations agree
with every independent-reference query cell across all 64 heads. Intermediate capture isolates
a one-ULP Metal cosine difference; frequency, sine and FMA counterfactuals do not fix that cell.
Native normal and compressed spectra now use resident FP32 complex-factor tables, computed once
from FP32 source-style angles, rather than per-head GPU transcendental calls. The tables add
512 MiB at maximum context, included in admission and residency, not reported as weight bytes.
There is no per-request factor allocation, transfer or CPU trigonometry. Retained-history replay
now matches every attention-stage BF16 cell and all 13 isolated FP32 projections, with zero
execution disk reads and page-ins. All 59 Metal and 10 reference checks pass. This retained-case
proof does not establish CUDA factor parity at every long-context position or clear the failing
eight-token whole-model gate; its oracle logits remain bit-identical to the previous candidate.

The resident-factor full eight-token trace preserves every native logit bit, but fails zero-I/O
qualification with 61,440 process disk-read bytes and zero page-ins. The corrected token-5
layer-0 attention now matches. Token-6 layer-0 MoE replay isolates two one-ULP routing-weight
differences with identical normalized inputs, logits and expert IDs. Expert-146 activation cell
477 is native 0.0810546875 versus wide-oracle 0.08056640625; exact pinned Torch CPU and CUDA
activation tails agree with native at every tested BF16 cell. That single cell propagates to
1,614 expert down-projection cells and 246 combined-branch cells. The diagnostic using native
weights matches every expert and branch stage, but never replaces the qualification oracle.
This replay measures zero disk reads and page-ins; no source-backed native correction is justified.
A separate complex-product cancellation probe agrees between
native Metal and exact pinned Torch CPU/CUDA: the real result is 2^-24, not zero. Disabling
complex-product contraction would therefore be an unjustified change.

A broader exact-source factor probe covers positions 0–7, all 32 pairs, both configured spectra.
Reference FP32 unaries and native host-FP64 unaries rounded to FP32 agree across all 1,024
components, so changing the reference unary width is not justified. Canonical CUDA differs
in 57 factor components; captured power, YaRN ramp and same-angle polar operations expose
device rounding differences. Both factor sets still give identical BF16 outputs on the retained
token-5 query. The table correction is therefore a proven retained-consumer fix, not a claim
of universal source-CUDA factor identity.

The minimal native API campaign used one uncached greedy request per case, thinking off, 16 output tokens,
on an M3 Ultra with 512 GiB RAM. Every stream completed with the exact requested prompt/output token counts
and zero cached tokens. Decode excludes the first token: 15 divided by the first-to-last-token interval.
These are single-run smoke measurements, not statistical estimates or a before/after speedup claim.

| Workload | Prompt tokens | TTFT (s) | Decode (tok/s) |
| --- | ---: | ---: | ---: |
| Prose | 128 | 10.57 | 1.46 |
| Prose | 512 | 10.71 | 1.25 |
| Prose | 1,024 | 23.23 | 1.17 |
| Prose | 2,048 | 50.00 | 1.15 |
| Prose | 4,096 | 103.93 | 1.12 |
| Code | 4,096 | 105.68 | 1.11 |

A separate long-prefill run passed 128k tokens with zero observed process disk-read and page-in deltas.
It was intentionally cancelled before generation; million-token execution remains unqualified.
