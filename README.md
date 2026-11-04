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

## Preserved history

`qwen35-historical-receipts-20261104.tar.gz` preserves all 53 original source-tree receipt files before condensing. Historical failures and actual producer identities remain unchanged.
The three `20261103` evidence files are unchanged from commit `9d07324dc7920fcff03152ff48c8fcc0e408a425`. They preserve the earlier reference experiment and the full CUDA run that exited 1 with 342 launch-limit problems; they do not qualify the current binary.

## Publication review

The current 387-file payload, identifying JSON fields, full logs, numeric-capture provenance, archive headers and outgoing contribution history were reviewed before publication. All 11 actual BF16 shard hashes/sizes match the pinned public LFS bytes and were reverified after serving. Numeric arrays remain byte-identical.
Private infrastructure identifiers, user paths, contacts, credentials and application-specific objectives are excluded. Generated prose samples are omitted and client interval origins normalized; durations, rates, fingerprints and numerical failures are retained. Generic container mounts and public software/hardware identities remain.
Archive owner/group names are empty and IDs/timestamps zero. Every extracted member is hash-verified. This review covers the submitted publication; earlier cached copies outside it have their own scope.

`SHA256SUMS` pins all six files. PR #2020 retains summary receipts and reproduction commands, with immutable links to these files.
