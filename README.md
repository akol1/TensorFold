# Public Qwen BF16 qualification evidence

Companion assets for ashhart/TensorFold PR #2020, kept outside its source branch at the maintainer's request.
The checkpoint and fixtures are public. All 21 archive members and both additional text payloads were reviewed for identifying metadata and secrets before publication. Host identity is redacted; recorded paths are generic container paths. SHA256SUMS identifies the exact bytes; the source PR's asset.json also lists every archive member checksum.

The Transformers payload and log preserve the original experiment, including its limitations. A completed experiment is not a passing qualification. The measured BF16-distance comparison has failing captured rows; the source PR retains the detailed verdict and remains draft. The complete CUDA contract exits 1 with 342 launch-limit problems. These files are evidence, not a software release.

Original native serving build: 327ef639fb7dbc355f44b4a5efc53209a3e06543.
Original strict contract source: 5332d97eb387af7e1a10433d77ee032a86bc631a.
Original public receipt publication: b651e8b411d62ece1d25fd33d4bbbc6edcbb6075.
