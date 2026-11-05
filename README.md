# Public Qwen BF16 qualification evidence

Companion evidence for ashhart/TensorFold PR #2020. These are data and logs, not a software release.
The pinned public Qwen checkpoint, public/synthetic fixtures, source identities and exact binaries are recorded in the manifests. No weights are distributed.

## Current evidence

`qwen35-bf16-readiness-20261104.tar.gz` contains the complete serving/device receipt, exact inputs, per-request results, independent reference/capture reports, resource audits and full CUDA log.
`qwen35-bf16-readiness-20261104.manifest.json` records every current and historical archive member's size and SHA-256, the archive hashes, and publication-review scope.

Actual GPU source: `b168b36d0cc3503ae69e8bffcad494574019caf5`. Pinned-main baseline: `0d19cc19b5b49a0b4a0188a56b6705bbedae91e4`.
The public-alias equivalent source preserves the same tree. Later parent receipts/host tests, an optional CPU dependency guard and condensed documentation do not relabel device executions.

Complete CUDA: **559 kernels pass gates 1-4, zero fail; 367/367 controls red as planted; zero launch-limit problems; exit 0**. The same six existing known findings reproduce.
Served API and same-weight FP64 checks pass. The direct HTTP-versus-Transformers experiment covers all 4511 scored positions, but strict distances fail in both directions. No wider bound replaces those failures.
The format remains an experimental functional text/scoring contribution; model equivalence, production and performance qualification are not claimed. BF16 prefill/concurrency remain slow. Other physical chips/backends and multiple ranks remain unqualified.

## Per-length comparison supplement

`qwen35-bf16-per-length-20261104.json` reports worst and median absolute served-minus-FP32 error over every scored position of each prompt length, alongside that length's own worst BF16-minus-FP32 error. The simple per-length worst-error rule fails at all four lengths. The stricter row-level diagnostics remain in the complete payload.

Run `python3 summarize-qwen35-served-reference.py qualified/served-reference.json` after unpacking the current archive to reproduce the supplement. It is derived from the exact checksum-pinned published member; no extra allowance is added.

## Preserved history

`qwen35-historical-receipts-20261104.tar.gz` preserves all 53 original source-tree receipt files before condensing. Historical failures and actual producer identities remain unchanged.
The three `20261103` evidence files are unchanged from commit `9d07324dc7920fcff03152ff48c8fcc0e408a425`. They preserve the earlier reference experiment and the full CUDA run that exited 1 with 342 launch-limit problems; they do not qualify the current binary.

## Publication review

The current 387-file payload, identifying JSON fields, full logs, numeric-capture provenance, archive headers and outgoing contribution history were reviewed before publication. All 11 actual BF16 shard hashes/sizes match the pinned public LFS bytes and were reverified after serving. Numeric arrays remain byte-identical.
Private infrastructure identifiers, user paths, contacts, credentials and application-specific objectives are excluded. Generated prose samples are omitted and client interval origins normalized; durations, rates, fingerprints and numerical failures are retained. Generic container mounts and public software/hardware identities remain.
Archive owner/group names are empty and IDs/timestamps zero. Every extracted member is hash-verified. This review covers the submitted publication; earlier cached copies outside it have their own scope.

`SHA256SUMS` pins all eight files. PR #2020 retains summary receipts and reproduction commands, with immutable links to these files.


## 20261105 layer diagnostic

The 32-input-token fixture is captured at embedding, all 64 residual boundaries and final normalization. The primary rule covers the 31 hidden rows used by served prompt scores; all 32 input rows are retained separately.

The first exceeding residual is zero-based **layer 0**: native-to-FP32 worst **0.07647705078125**, against Transformers BF16-to-FP32 worst **0.056326866149902344**. Embeddings match exactly. The simple maximum rule fails at 18/64 residual layers. Final normalization falls inside its own hidden-state maximum, which does not waive the separately failing served-logprob comparison. No tolerance is changed, and this does not establish a single faulty accumulation or a format floor.

Extract all three `qwen35-bf16-layers-*-20261105.tar.gz` archives into the same directory. The native archive includes the shared metadata, exact IDs, complete comparison, build/capture logs and source identities. The other two archives hold the 66 BF16-arm and 66 promoted-FP32-arm arrays. The manifest checks every archive/member and binds the comparison script to its source.

Native capture producer: `9416f0cf94bddbea49becb4c14047d0c16d08faf`; Transformers capture producer: `b54d8ff7691385fe420399e72272c932a881e29e`; served-row comparison: `c6930073d5192663bd5726ff74a8fa426f7a06aa`; main base: `002c9e03eecf9b10b83a73b9149dc7f49d50368a`. Capture passes its traced/untraced logit identity check. Both reference arms pass their independent cross-checks, preserve all 96 stored FP32 parameters, and all 11 checkpoint shard hashes match pinned public LFS after capture. Original running deployments and active-service health were restored.

The final-source plain build and five-SM CPU resource audit pass. The latest complete reference test suite passes 20/20 with its ML dependencies; the superseded partial-export setup errors remain in the packet.

This supplement is a diagnostic on one sm_120 GPU with CPU references. It contains no new performance run or whole-model-equivalence claim. The earlier complete CUDA and serving receipts retain their actual older producer; a complete device suite was not rerun on the rebased head.

The 217-file supplement, including 198 numeric tensors, was reviewed for private identifiers and credentials. Tensor bytes are unchanged; archive identity fields are cleared, and the supplemental host log normalizes local checkout/user prefixes. Public contributor aliases and numerical failures are retained.
