//! Engram hash, FP8 gather and signed-sqrt gate at the FP32 contract and the FP64 ideal; receipts print "ENG64".
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");

const engram = tf.deepseek_v41_engram;
const hash = tf.deepseek_v41_engram_hash;
const View = engram.View;

const dim: usize = 5120;
const copies: usize = 4;
const hidden_width: usize = dim * copies;
const kv_width: usize = dim * (copies + 1);
const gather_dim: usize = engram.gather_dim;
const head_dim: usize = engram.head_dim;
const eps: f32 = 1e-6;
const raw_ids: usize = 64;

/// Synthetic token map and history so the hash case needs no checkpoint bytes.
const hash_tokens: usize = 10;
const hash_chunk: usize = 8;

/// One gather case covers every column of the real table layout with fewer rows.
const gather_tokens: usize = 4;
const table_rows: usize = 128;

/// Two gate rows: the second carries a DEAD mask so the zero-gate path is checked exactly.
const gate_tokens: usize = 2;

const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    fn next(self: *Lcg) u64 {
        self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
        return self.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    fn uniform(self: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(self.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
    fn below(self: *Lcg, limit: usize) usize {
        return @intCast(self.next() % limit);
    }
};

const Delta = struct {
    cells: usize = 0,
    unequal: usize = 0,
    nrms: f64 = 0,
    max_abs: f64 = 0,
    worst: usize = 0,
};

fn deltaBf16(actual: []const u16, ideal: []const u16) Delta {
    var d: Delta = .{ .cells = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |a, want, i| {
        const ref: f64 = math.d64(want);
        if (std.math.isFinite(ref)) magnitude += ref * ref;
        if (a == want) continue;
        const got: f64 = math.d64(a);
        const diff = got - ref;
        d.unequal += 1;
        if (@abs(diff) > d.max_abs) {
            d.max_abs = @abs(diff);
            d.worst = i;
        }
        if (std.math.isFinite(diff)) squared += diff * diff;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

fn printBf16(stage: []const u8, case_name: []const u8, d: Delta) void {
    std.debug.print("ENG64 {{\"stage\":\"{s}\",\"case\":\"{s}\",\"cells\":{d},\"bf16_unequal\":{d},\"nrms\":{d},\"max_abs\":{d},\"worst\":{d}}}\n", .{ stage, case_name, d.cells, d.unequal, d.nrms, d.max_abs, d.worst });
}

fn printEq(stage: []const u8, case_name: []const u8, cells: usize, unequal: usize) void {
    std.debug.print("ENG64 {{\"stage\":\"{s}\",\"case\":\"{s}\",\"cells\":{d},\"metal_vs_contract_unequal\":{d}}}\n", .{ stage, case_name, cells, unequal });
}

fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

fn completed(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.EngramGpuFailure;
}

/// The pinned CPU oracle's gate: wide sums, the FP32 mean projection, then FP32 rstd, dot and gate.
fn contractProduct(value: f32, scale: f32) u16 {
    const product = value * scale;
    if (@abs(scale) < 0x1p-126 or @abs(product) < 0x1p-126) return 0;
    return math.b(product);
}

fn contractGate(hidden: []const u16, kv: []const u16, q: []const f32, k: []const f32, copy: usize, out: []u16) void {
    @setFloatMode(.strict);
    var hh: f64 = 0;
    var kk: f64 = 0;
    var dot: f64 = 0;
    for (0..dim) |d| {
        const at = copy * dim + d;
        const h = math.f(hidden[at]);
        const x = math.f(kv[at]);
        const weight = q[at] * k[at];
        hh += @as(f64, h) * @as(f64, h);
        kk += @as(f64, x) * @as(f64, x);
        dot += @as(f64, (h * weight) * x);
    }
    const h_inverse = math.idealRsqrt(math.meanFromWide(hh, dim) + eps);
    const k_inverse = math.idealRsqrt(math.meanFromWide(kk, dim) + eps);
    const rstd = h_inverse * k_inverse;
    const dot_wide: f32 = @floatCast(dot);
    const normalized = (dot_wide * rstd) * (1 / @sqrt(@as(f32, 5120.0)));
    const root = std.math.copysign(@sqrt(@max(@abs(normalized), @as(f32, 1e-6))), normalized);
    const gate = math.sigmoid(root);
    for (0..dim) |d| {
        const at = copy * dim + d;
        out[d] = math.b(math.f(hidden[at]) + gate * math.f(kv[hidden_width + d]));
    }
}

/// The same equations widened to FP64, never an emulation of the Metal kernel.
fn idealGate(hidden: []const u16, kv: []const u16, q: []const f32, k: []const f32, copy: usize, out: []u16) void {
    @setFloatMode(.strict);
    var hh: f64 = 0;
    var kk: f64 = 0;
    var dot: f64 = 0;
    for (0..dim) |d| {
        const at = copy * dim + d;
        const h = math.d64(hidden[at]);
        const x = math.d64(kv[at]);
        const weight = @as(f64, q[at]) * @as(f64, k[at]);
        hh += h * h;
        kk += x * x;
        dot += (h * weight) * x;
    }
    const h_inverse = 1 / @sqrt(hh / 5120.0 + @as(f64, eps));
    const k_inverse = 1 / @sqrt(kk / 5120.0 + @as(f64, eps));
    const normalized = (dot * (h_inverse * k_inverse)) / @sqrt(@as(f64, 5120.0));
    const root = std.math.copysign(@sqrt(@max(@abs(normalized), @as(f64, 1e-6))), normalized);
    const gate = 1 / (1 + @exp(-root));
    for (0..dim) |d| {
        const at = copy * dim + d;
        out[d] = math.b64(math.d64(hidden[at]) + gate * math.d64(kv[hidden_width + d]));
    }
}

fn sampleBf16(lcg: *Lcg, family: usize, i: usize) u16 {
    const scale: f32 = switch (family) {
        0 => 1,
        1 => 16,
        2 => 0.0078125,
        else => if (i % 2 == 0) 1 else 0.0009765625,
    };
    return math.b(lcg.uniform() * scale);
}

const family_names = [_][]const u8{ "unit", "wide", "narrow", "cancellation" };

test "Engram hash columns equal the integer contract across a masked chunk and committed history" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try engram.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var lcg = Lcg{};
    var map_bytes: [raw_ids * 4]u8 = undefined;
    for (0..raw_ids) |i| std.mem.writeInt(u32, map_bytes[i * 4 ..][0..4], @intCast(500 + lcg.below(90000)), .little);
    const map = try device.buffer(map_bytes.len, 0);
    defer map.deinit();
    @memcpy(map.slice(u8, map_bytes.len), &map_bytes);
    const chunk_ids = [_]u32{ 7, 9, 11, 3, 13, 15, 17, 19, 21, 23 };
    const alive = [_]u8{ 1, 1, 1, 0, 1, 1, 1, 1, 1, 1 };
    const ids = try device.buffer(hash_tokens * 4, 0);
    defer ids.deinit();
    @memcpy(ids.slice(u32, hash_tokens), &chunk_ids);
    const mask = try device.buffer(hash_tokens, 0);
    defer mask.deinit();
    @memcpy(mask.slice(u8, hash_tokens), &alive);
    const history = try device.buffer(engram.history_bytes, 0);
    defer history.deinit();
    const first = try device.buffer(hash_tokens * hash.columns * 4, 0);
    defer first.deinit();
    const second = try device.buffer(hash_tokens * hash.columns * 4, 0);
    defer second.deinit();
    const cb = queue.commandBuffer();
    var e = cb.compute(.serial);
    engram.encodeReset(e, &pipelines, view(history));
    for (0..2) |layer| {
        const out = if (layer == 0) first else second;
        try engram.encodeHash(e, &pipelines, hash.layers[layer], view(ids), view(map), view(mask), view(history), view(out), @intCast(hash_chunk));
    }
    try engram.encodeCommit(e, &pipelines, view(ids), view(map), view(mask), view(history), @intCast(hash_chunk));
    for (0..2) |layer| {
        const out = if (layer == 0) first else second;
        try engram.encodeHash(e, &pipelines, hash.layers[layer], view(ids).shifted(hash_chunk * 4), view(map), view(mask).shifted(hash_chunk), view(history), view(out).shifted(hash_chunk * hash.columns * 4), @intCast(hash_tokens - hash_chunk));
    }
    e.end();
    try completed(cb);
    var expected: [2][hash_tokens][hash.columns]u32 = undefined;
    for (0..2) |layer| {
        var cpu: hash.History = .{};
        try hash.hashChunk(&cpu, .{ .bytes = &map_bytes }, hash.layers[layer], &chunk_ids, &alive, &expected[layer]);
    }
    var unequal: usize = 0;
    for (0..2) |layer| {
        const out = if (layer == 0) first else second;
        const got = out.slice(u32, hash_tokens * hash.columns);
        for (expected[layer], 0..) |row, t| {
            for (row, 0..) |want, col| {
                if (got[t * hash.columns + col] != want) unequal += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), unequal);
    printEq("hash", "masked-chunk-and-history-carry", 2 * hash_tokens * hash.columns, unequal);
}

test "Engram gather narrows the E4M3 row grid to BF16 exactly like the FP32 contract" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try engram.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var lcg = Lcg{};
    var weight_bytes: [table_rows * head_dim]u8 = undefined;
    var scale_bytes: [table_rows * (head_dim / 32)]u8 = undefined;
    for (&weight_bytes) |*b| b.* = @intCast(lcg.below(127));
    for (&scale_bytes) |*b| b.* = @intCast(lcg.below(255));
    const weight = try device.buffer(weight_bytes.len, 0);
    defer weight.deinit();
    @memcpy(weight.slice(u8, weight_bytes.len), &weight_bytes);
    const scale = try device.buffer(scale_bytes.len, 0);
    defer scale.deinit();
    @memcpy(scale.slice(u8, scale_bytes.len), &scale_bytes);
    const row_ids = try device.buffer(gather_tokens * hash.columns * 4, 0);
    defer row_ids.deinit();
    const row_slice = row_ids.slice(u32, gather_tokens * hash.columns);
    for (row_slice) |*id| id.* = @intCast(lcg.below(table_rows));
    const gathered = try device.buffer(gather_tokens * gather_dim * 2, 0);
    defer gathered.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try engram.encodeGather(e, &pipelines, .{ .weight = view(weight), .scale = view(scale), .rows = @intCast(table_rows) }, view(row_ids), view(gathered), @intCast(gather_tokens), gather_dim);
    e.end();
    try completed(cb);
    const got = gathered.slice(u16, gather_tokens * gather_dim);
    var contract: [gather_tokens * gather_dim]u16 = undefined;
    var ideal: [gather_tokens * gather_dim]u16 = undefined;
    for (0..gather_tokens) |t| {
        for (0..hash.columns) |col| {
            for (0..head_dim) |d| {
                const row: usize = row_slice[t * hash.columns + col];
                const code = weight_bytes[row * head_dim + d];
                const block = scale_bytes[row * (head_dim / 32) + d / 32];
                contract[t * gather_dim + col * head_dim + d] = contractProduct(math.e4(code), math.e8(block));
                ideal[t * gather_dim + col * head_dim + d] = math.b64(@as(f64, math.e4(code)) * @as(f64, math.e8(block)));
            }
        }
    }
    var unequal: usize = 0;
    for (got, contract[0..]) |a, want| {
        if (a != want) unequal += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), unequal);
    printEq("gather", "fp8-row-grid", got.len, unequal);
    printBf16("gather", "metal-vs-fp64-ideal", deltaBf16(got, &ideal));
    printBf16("gather", "fp32-contract-vs-fp64-ideal", deltaBf16(&contract, &ideal));
}

test "Engram gate reaches BF16 at the FP32 contract over matching equations" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try engram.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var lcg = Lcg{};
    var q: [hidden_width]f32 = undefined;
    var k: [hidden_width]f32 = undefined;
    for (&q, &k) |*q_value, *k_value| {
        q_value.* = lcg.uniform();
        k_value.* = lcg.uniform();
    }
    var hidden: [gate_tokens * hidden_width]u16 = undefined;
    var kv: [gate_tokens * kv_width]u16 = undefined;
    var ideal_row: [gate_tokens * hidden_width]u16 = undefined;
    var contract_block: [dim]u16 = undefined;
    const q_weight = try device.buffer(hidden_width * 4, 0);
    defer q_weight.deinit();
    @memcpy(q_weight.slice(f32, hidden_width), &q);
    const k_weight = try device.buffer(hidden_width * 4, 0);
    defer k_weight.deinit();
    @memcpy(k_weight.slice(f32, hidden_width), &k);
    const hidden_buffer = try device.buffer(hidden.len * 2, 0);
    defer hidden_buffer.deinit();
    const kv_buffer = try device.buffer(kv.len * 2, 0);
    defer kv_buffer.deinit();
    const gate_mask = try device.buffer(gate_tokens, 0);
    defer gate_mask.deinit();
    for (gate_mask.slice(u8, gate_tokens), 0..) |*m, i| m.* = @intFromBool(i == 0);
    const output = try device.buffer(gate_tokens * hidden_width * 2, 0);
    defer output.deinit();
    const masked_output = try device.buffer(gate_tokens * hidden_width * 2, 0);
    defer masked_output.deinit();
    const gate = engram.Gate{
        .q_weight = view(q_weight),
        .k_weight = view(k_weight),
        .dim = @intCast(dim),
        .copies = @intCast(copies),
        .hidden_stride = hidden_width,
        .kv_stride = kv_width,
        .output_stride = hidden_width,
        .eps = eps,
    };
    var contract_unequal: usize = 0;
    for (family_names, 0..) |family, index| {
        for (&hidden, 0..) |*v, i| v.* = sampleBf16(&lcg, index, i);
        for (&kv, 0..) |*v, i| v.* = sampleBf16(&lcg, index, i + 7);
        @memcpy(hidden_buffer.slice(u16, hidden.len), &hidden);
        @memcpy(kv_buffer.slice(u16, kv.len), &kv);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try engram.encodeGate(e, &pipelines, gate, view(hidden_buffer), view(kv_buffer), null, view(output), @intCast(gate_tokens));
        e.end();
        try completed(cb);
        const got = output.slice(u16, gate_tokens * hidden_width);
        for (0..gate_tokens) |t| {
            for (0..copies) |copy| {
                const hidden_row = hidden[t * hidden_width ..][0..hidden_width];
                const kv_row = kv[t * kv_width ..][0..kv_width];
                contractGate(hidden_row, kv_row, &q, &k, copy, &contract_block);
                idealGate(hidden_row, kv_row, &q, &k, copy, ideal_row[t * hidden_width + copy * dim ..][0..dim]);
                for (0..dim) |d| {
                    const at = t * hidden_width + copy * dim + d;
                    if (got[at] != contract_block[d]) contract_unequal += 1;
                }
            }
        }
        printBf16("gate", family, deltaBf16(got, &ideal_row));
    }
    try std.testing.expect(contract_unequal <= (gate_tokens * hidden_width) / 100);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try engram.encodeGate(e, &pipelines, gate, view(hidden_buffer), view(kv_buffer), view(gate_mask), view(masked_output), @intCast(gate_tokens));
    e.end();
    try completed(cb);
    const masked = masked_output.slice(u16, gate_tokens * hidden_width);
    try std.testing.expectEqualSlices(u16, hidden[hidden_width..][0..hidden_width], masked[hidden_width..][0..hidden_width]);
    printEq("gate", "all-families-metal-vs-contract", gate_tokens * hidden_width, contract_unequal);
}
