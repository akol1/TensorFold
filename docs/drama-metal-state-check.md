# Drama Metal structural state checker

Build without running inference:

```sh
zig build drama3-metal-chunk-check test-drama-metal-chunk-host -Dcpu=apple_m1
```

Run on an otherwise idle Metal device, using the checkpoint and raw native prompt rows under test:

```sh
zig-out/bin/drama3-metal-chunk-check --model MODEL --rows prompt.i32 --context 8192 --chunk 128
```

`prompt.i32` contains native-endian signed int32 rows of width 11, as emitted by `drama3-generate --rows-out`. The checker requires at least two rows and reserves two positions. It admits context only from 1025 through 8192. It does not raise serving limits. Both states share weights and a GPU queue, but have separate KV, recurrent and scratch allocations of the same capacity.

The reference processes one prompt row at a time; the candidate processes the requested chunk width. At each candidate chunk boundary, the checker compares exact live recurrent and KV bytes. It excludes unused cache tails and scalar padding. Final prompt hidden state, current input, sampling window and counters are compared separately. A mismatch reports its stage, layer and byte and exits nonzero. Width 1 is the control; use nonmultiples of the chunk width to exercise tails.

## Optional forced-history seam

```sh
zig-out/bin/drama3-metal-chunk-check --model MODEL --rows prompt.i32 --context 8192 \
  --chunk 128 --forced-rows acoustic.i32 --poison-kv 1
```

The forced file uses the same 11-column format and must contain valid acoustic rows. Prompt rows plus forced rows plus two reserved positions must fit. The checker compares prompt prefill followed by forced frame steps against prefill of the identical concatenated history. It checks live state and final hidden state before capturing a further identical forced step: candidate logits, all 13 tower layer outputs and nine residual-head logits, with explicit sequence and shape checks. It then compares post-step state and output and verifies the emitted frame equals the supplied row. The additional step repeats the last forced row; it is not sampled speech.

`--poison-kv 1` fills only the candidate's live KV region with a fixed pattern before full-history prefill. This checks that stale matching KV cannot mask missing writes. It requires `--forced-rows`; omit it for the separate normal run.

## Optional retained-KV resume

Add `--resume-cut N` for a nonzero chunk-aligned cut strictly before the penultimate prompt row. The checker captures recurrent state at that cut, completes the suffix to change that state, restores the snapshot, and replays the suffix against fresh execution. SHA256 integrity checks verify retained prefix KV across the operation. The cut preserves the tower's next-row lookahead. Snapshots do not store KV and this does not test persistence across process restart. The resume check precedes the optional forced-history check.

## Optional distinct-history cooperative isolation

Add `--other-rows other-prompt.i32 --other-forced-rows other-acoustic.i32` alongside `--forced-rows`. Both prompts and forced histories must differ. Each history must satisfy the same context and code-range checks before loading. This mode limits chunk width to 1..128 and checks the existing four-state memory-admission plan; it does not override the machine's load limit. That plan has a fixed per-state scratch allowance, not a chunk-dependent allocation or measured peak estimate. The chunk cap bounds scratch growth to the tested range; passing admission is not proof of physical footprint.

The checker runs each history alone, then alternates one prompt chunk or forced frame between two separate lanes under both AB and BA orders. Different lengths exercise one lane decoding while the other still prefills. All four states share weights and a queue, with separate mutable state and input buffers. KV is poisoned before each schedule. Each lane's terminal state and forced outputs must equal its own solo reference. This mode does not compare per-step isolation logits or exercise the HTTP scheduler, codec, cancellation or simultaneous GPU execution. A separate negative control should copy a verified-different recurrent-state region from one lane into the other and fail with a lane-labelled mismatch.

## Interpretation and safety

These are zero-tolerance structural comparisons within one native implementation. They do not establish independent checkpoint fidelity, natural audio quality, CFG, stochastic sampler equivalence, concurrent scheduling or a wider context. Matching common-mode errors or nonfinite values are not excluded by byte equality. Readbacks synchronize the queue; these runs cannot establish asynchronous ordering correctness. Report each enabled gate, actual row counts, chunk width, checkpoint revision and binary identity. Comparison byte counts include repeatedly checked prefixes and are not a memory footprint measurement.

Use the machine owner's inference and fan-control policy. Do not change shared memory limits or stop unrelated servers. The tool loads weights and allocates two states, or four for cooperative isolation; build and host-test steps do not run it. Validate measurement with a negative control: a temporary recurrent-state bit flip must fail, and omitting resume restoration must fail. Never publish a fault-injected binary as the normal checker.
