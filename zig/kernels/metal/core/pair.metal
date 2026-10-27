// Two Macs' tensor-parallel exchanges: the host relays parts through shared events; these kernels only do arithmetic.
#include <metal_stdlib>
using namespace metal;

// Sync page words (shared with pair.zig): FAILED = the host gave up on a seam; OBSERVED = this side's refusals.
constant constexpr uint FAILED = 16, OBSERVED = 20;

// True once the host gave up: FAILED is set before the wait releases, so a later kernel poisons its output rows.
inline bool pair_link_broken(device atomic_uint* sync, uint t) {
  if (atomic_load_explicit(&sync[FAILED], memory_order_relaxed) == 0u) return false;
  if (t == 0) atomic_fetch_add_explicit(&sync[OBSERVED], 1u, memory_order_relaxed);
  return true;
}

inline float pair_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Kernels' argument block, byte-identical to pair.zig's PairRows: four u32 then four f32; no payload element.
struct PairRows {
  uint width, stride, rank, x;
  float eps, pad0, pad1, pad2;
};

// bf16 bits as the exact fp32 value they name (argmax reads a bf16 logits stream this way).
inline float pair_bf16(uint bits) {
  return as_type<float>(bits << 16);
}

// Row r: h += rank 0's part + rank 1's part, then y = w * (h / rms(h)); payload is fp32 (streamBytes).
kernel void pair_add_rms(device atomic_uint* sync [[buffer(0)]], constant PairRows& a [[buffer(1)]],
                         const device float* mine [[buffer(2)]], device atomic_uint* peer [[buffer(3)]],
                         device float* h [[buffer(4)]], const device bfloat* w [[buffer(5)]],
                         device float* y [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  if (pair_link_broken(sync, t)) {
    for (uint i = t; i < a.width; i += 256) y[ulong(r) * a.stride + i] = NAN;
    return;
  }
  device float* x = h + ulong(r) * a.stride;
  float s = 0.0f;
  for (uint i = t; i < a.width; i += 256) {
    const ulong at = ulong(r) * a.width + i;
    const float other = as_type<float>(atomic_load_explicit(&peer[at], memory_order_relaxed));
    const float sum = a.rank == 0 ? mine[at] + other : other + mine[at];
    const float v = x[i] + sum;
    x[i] = v;
    s = fma(v, v, s);
  }
  s = pair_lanes(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8; ++g) var += part[g];
  const float rs = precise::rsqrt(var / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += 256) y[ulong(r) * a.stride + i] = float(w[i]) * (x[i] * rs);
}

// Row r: this rank's own half of the frame row into the exchange slot; a bf16 copy, so it carries the frame's bits.
kernel void pair_stage(device atomic_uint* sync [[buffer(0)]], constant PairRows& a [[buffer(1)]],
                       const device ushort* src [[buffer(2)]], device ushort* dst [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const ulong in = ulong(r) * a.stride;
  const ulong out = ulong(r) * a.width;
  if (pair_link_broken(sync, t)) {
    for (uint i = t; i < a.width; i += 256) dst[out + i] = 0x7FC0u; // bf16 NaN: the link stays down
    return;
  }
  for (uint i = t; i < a.width; i += 256) dst[out + i] = src[in + i];
}

// Row r: this rank's half at its own offset in the frame row and the peer's at the other; a bf16 copy, no arithmetic.
kernel void pair_gather(device atomic_uint* sync [[buffer(0)]], constant PairRows& a [[buffer(1)]],
                        const device ushort* mine [[buffer(2)]], device atomic_uint* peer [[buffer(3)]],
                        device ushort* out [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
  const ulong mrow = ulong(r) * a.width;
  const ulong orow = ulong(r) * a.stride;
  const ulong own = a.rank == 0 ? orow : orow + a.width;
  const ulong got = a.rank == 0 ? orow + a.width : orow;
  if (pair_link_broken(sync, t)) {
    for (uint i = t; i < a.width; i += 256) {
      out[own + i] = 0x7FC0u;
      out[got + i] = 0x7FC0u;
    }
    return;
  }
  for (uint i = t; i < a.width; i += 256) {
    // The relay wrote the peer's half: an untracked frame read, so one atomic word (two bf16) at a time.
    const uint word = atomic_load_explicit(&peer[(mrow + i) >> 1], memory_order_relaxed);
    out[own + i] = mine[mrow + i];
    out[got + i] = ushort(word >> ((i & 1) * 16));
  }
}

// Row r's argmax over this rank's logits (ids lo..lo+n): (fp32 value, id), lower id on tie; dims.w 0 = fp32, 1 = bf16.
kernel void pair_argmax(const device float* logits [[buffer(0)]], constant uint4& dims [[buffer(1)]],
                        device uint* out [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = ulong(r) * dims.x;
  const device float* xf = logits + row;
  const device ushort* xb = reinterpret_cast<const device ushort*>(logits) + row;
  const bool wide = dims.w != 0u;
  float best = -INFINITY;
  uint at = 0xFFFFFFFFu;
  for (uint i = t; i < dims.z; i += 1024) {
    const float v = wide ? pair_bf16(uint(xb[i])) : xf[i];
    if (v > best) best = v, at = dims.y + i;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  threadgroup float vb[32];
  threadgroup uint va[32];
  if (lane == 0) vb[sg] = best, va[sg] = at;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vb[lane];
  at = va[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) best = ob, at = oa;
  }
  if (lane == 0) out[2 * r] = as_type<uint>(best), out[2 * r + 1] = at;
}

// Each row's pick over both Macs' argmaxes: the larger value, the lower id on a tie (one Mac's argmax).
kernel void pair_pick(device atomic_uint* sync [[buffer(0)]], constant uint2& rx [[buffer(1)]],
                      const device uint* mine [[buffer(2)]], device atomic_uint* peer [[buffer(3)]],
                      device uint* picks [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  if (pair_link_broken(sync, t)) {
    if (t == 0) for (uint r = 0; r < rx.x; ++r) picks[r] = 0xFFFFFFFFu;  // no such token: visibly wrong, never a token
    return;
  }
  if (t != 0) return;
  for (uint r = 0; r < rx.x; ++r) {
    const float vm = as_type<float>(mine[2 * r]);
    const uint am = mine[2 * r + 1];
    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
  }
}
