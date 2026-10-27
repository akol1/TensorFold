//! The pair seam's five kernels (core/pair.metal): what pair.zig launches and the float64 rules the gates check.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows_k = contract.rows;

/// The seam's five kernels; the tail of each registry name is the MSL symbol launchOf hands the source.
pub const Which = enum { add_rms, stage, gather, argmax, pick };

pub const all = [_]Which{ .add_rms, .stage, .gather, .argmax, .pick };

/// What pair.zig dispatches per kernel, so `init` can compare it against Metal's own reflection.
pub const Launched = struct { threads: u32, tg_memory: u32 };

pub fn launch(w: Which) Launched {
    return switch (w) {
        .add_rms => .{ .threads = 256, .tg_memory = 32 },
        .stage => .{ .threads = 256, .tg_memory = 0 },
        .gather => .{ .threads = 256, .tg_memory = 0 },
        .argmax => .{ .threads = 1024, .tg_memory = 256 },
        .pick => .{ .threads = 32, .tg_memory = 0 },
    };
}

/// The MSL symbol pair.metal defines.
pub fn symbol(w: Which) []const u8 {
    return switch (w) {
        .add_rms => "pair_add_rms",
        .stage => "pair_stage",
        .gather => "pair_gather",
        .argmax => "pair_argmax",
        .pick => "pair_pick",
    };
}

/// The spec's name: `metal <what> <symbol>`, so launchOf's last word is the symbol and its prefix picks core_pair.
pub fn name(w: Which) []const u8 {
    return switch (w) {
        .add_rms => "metal pair add_rms pair_add_rms",
        .stage => "metal pair stage pair_stage",
        .gather => "metal pair gather pair_gather",
        .argmax => "metal pair argmax pair_argmax",
        .pick => "metal pair pick pair_pick",
    };
}

/// Qwen3.8-27B's rank-local slice: the seam reduces ids [124160, 248320) on rank 1.
const logit_lo: u32 = 124160;

/// The RMS epsilon both callers pass (qwen27's frame and glm_dsa's 1e-5).
const eps: f32 = 1e-5;

/// A thread's chain over every 256th square, five shuffle levels, then the eight partials.
fn chainSquares(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

/// A thread's chain over every 1024th logit, five shuffle levels, then the 32 simdgroup winners.
fn chainLogits(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 5 + 5;
}

/// Only moves and compares values: no sums to carry an accumulator's error.
fn noSums(_: c.Shape) usize {
    return 0;
}

const add_why = "y follows the sum of squares as ss^-1/2 and stays fp32: a bf16 sum's error is 2^-8 of the norm";
const move_why = "a move writes each bf16 element as it read it: no sum, no rounding, so no accumulator's error can enter";

const argmax_rule = "a winner is (value bits, id): NaN loses every comparison, so a row of only NaNs reports the -inf sentinel with id 0xFFFFFFFF";
const pick_rule = "the pick copies one rank's (value, id) word pair; an id equal to 0xFFFFFFFF is how a broken link reads";

/// A kernel's spec. The raw kernels check themselves; the two moves' bound is exact, so gate 3 keeps every edge.
pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .add_rms => .{
            .name = name(w),
            .accum = .f32,
            .depth = chainSquares,
            .ops = 8,
            .mids = 1,
            .out = .f32,
            .in_place = true,
            .tiles = .{ .{}, .{}, .{ .size = 256, .step = 256, .min = 256 } },
            .served = &.{
                .{ .rows = 8, .n = 5120, .k = 5120 },
                .{ .rows = 8, .n = 6144, .k = 6144 },
                .{ .rows = 2, .n = 256, .k = 256 },
            },
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = add_why },
        },
        // One spec for both moves; a gather's `n` is its frame row stride, and the 17,408 shape is seam B's.
        .stage, .gather => .{
            .name = name(w),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 256, .step = 256, .min = 256 } },
            .served = if (w == .stage) &.{.{ .rows = 8, .n = 5120, .k = 5120 }} else &.{
                .{ .rows = 8, .n = 10240, .k = 5120 }, .{ .rows = 8, .n = 17408, .k = 8704 },
                .{ .rows = 8, .n = 10496, .k = 5120 },
            },
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = move_why },
        },
        .argmax => .{
            .name = name(w),
            .accum = .none,
            .depth = chainLogits,
            .out = .raw,
            .nan = .{ .stated = argmax_rule },
            .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 64, .min = 64 } },
            .served = &.{.{ .rows = 8, .n = 1, .k = 124160 }},
        },
        .pick => .{
            .name = name(w),
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = pick_rule },
            .served = &.{ .{ .rows = 32, .n = 1, .k = 1 }, .{ .rows = 128, .n = 1, .k = 1 } },
        },
    };
}

/// `constant PairRows&` at buffer 1: `width` then the frame row `stride`; argmax and pick take their own words.
const PairRows = extern struct {
    width: u32,
    stride: u32,
    rank: u32,
    x: u32,
    eps: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

fn rowsArgs(width: usize, stride: usize, rank: u32, x: u32, e: f32) PairRows {
    return .{ .width = @intCast(width), .stride = @intCast(stride), .rank = rank, .x = x, .eps = e, .pad0 = 0, .pad1 = 0, .pad2 = 0 };
}

fn bit(v: f32) u32 {
    return @bitCast(v);
}

fn floatOf(v: u32) f32 {
    return @bitCast(v);
}

/// One prepared call: the buffers the kernel is bound to, the host values its reference reads, and its own check.
pub const Pair = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    /// The hooks this probe exposes: the raw kernels check their own rule, the three computing ones use the reference.
    v: g.VTable,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    rows: usize = 0,
    width: usize = 0,
    /// The rank this prepare writes first: the gather's `mine` half is its payload.
    rank: u32 = 0,
    /// The row stride in PairRows, in bf16 elements: the stage's source frame row, the gather's destination row.
    stride: usize = 0,

    // Host inputs of the last prepare, as the reference and the check read them (freed by `drop`).
    xh: ?[]f32 = null, // add_rms: this rank's part
    ph: ?[]f32 = null, // add_rms: the peer's part
    wh: ?[]u16 = null, // add_rms: the RMS weights, bf16
    hh: ?[]f32 = null, // add_rms: h before the launch (the kernel folds into it in place)
    sh: ?[]u16 = null, // stage: the src payload, bf16
    mh: ?[]u16 = null, // gather: this rank's half (mine), bf16
    nh: ?[]u16 = null, // gather: the peer's half (recv), bf16
    kh: ?[]u16 = null, // argmax: the bf16 logits
    pw: ?[]u32 = null, // pick: this rank's (value bits, id) words
    qw: ?[]u32 = null, // pick: the peer's words

    msg: [192]u8 = undefined,
    sync: rig_.Ref = undefined,
    mine: rig_.Ref = undefined,
    peer: rig_.Ref = undefined,
    h: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    logits: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Pair {
        const pipe = try rig_.pipeline(device, sources.core_pair, symbol(which));
        errdefer pipe.deinit();
        const want = launch(which);
        const tg = rig_.tgMemory(pipe);
        if (tg != want.tg_memory) {
            std.debug.print("metal pair: {s} takes {d} bytes of threadgroup memory, its record says {d}\n", .{ symbol(which), tg, want.tg_memory });
            return error.ThreadgroupMemoryMismatch;
        }
        var v = g.hooks(Pair);
        if (which != .argmax and which != .pick) v.check = null;
        if (which != .stage and which != .gather) v.bounds = null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .pipe = pipe, .v = v };
    }

    pub fn deinit(self: *Pair) void {
        self.drop();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Pair) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &self.v };
    }

    fn drop(self: *Pair) void {
        if (self.xh) |s| self.a.free(s);
        if (self.ph) |s| self.a.free(s);
        if (self.wh) |s| self.a.free(s);
        if (self.hh) |s| self.a.free(s);
        if (self.sh) |s| self.a.free(s);
        if (self.mh) |s| self.a.free(s);
        if (self.nh) |s| self.a.free(s);
        if (self.kh) |s| self.a.free(s);
        if (self.pw) |s| self.a.free(s);
        if (self.qw) |s| self.a.free(s);
        self.xh = null;
        self.ph = null;
        self.wh = null;
        self.hh = null;
        self.sh = null;
        self.mh = null;
        self.nh = null;
        self.kh = null;
        self.pw = null;
        self.qw = null;
    }

    /// Ordinary values: what the seam carries in production, symmetric about zero.
    fn ordinary(rng: std.Random) f32 {
        return rng.float(f32) * 2 - 1;
    }

    /// f32 values as bf16, for the buffers whose datum is a bf16 frame.
    fn framed(rng: std.Random) f32 {
        return c.fromBf16(c.toBf16(ordinary(rng)));
    }

    /// A bf16 payload: what the two moves carry, with gate 1's non-finite plants where the reference looks for them.
    fn payload(self: *Pair, len: usize, fill: g.Fill, rng: std.Random) ![]u16 {
        const s = try self.a.alloc(u16, len);
        for (s) |*v| v.* = c.toBf16(ordinary(rng));
        if (fill == .nonfinite) {
            const nan = c.toBf16(std.math.nan(f32));
            const inf = c.toBf16(std.math.inf(f32));
            const ninf = c.toBf16(-std.math.inf(f32));
            for ([_]u16{ nan, inf, ninf, nan, ninf }, 0..) |plant, p| {
                const at = (p + 1) * (len / 8 + 1) + p * 3;
                if (at < len) s[at] = plant;
            }
        }
        return s;
    }

    /// An input the seam would give: this rank's words, the peer's words, or the weights.
    fn part(self: *Pair, len: usize, bf16: bool, fill: g.Fill, rng: std.Random) ![]f32 {
        const s = try self.a.alloc(f32, len);
        for (s) |*v| v.* = if (bf16) framed(rng) else ordinary(rng);
        if (fill == .nonfinite) {
            const nan = std.math.nan(f32);
            const inf = std.math.inf(f32);
            for ([_]f32{ nan, inf, -inf, nan, -inf }, 0..) |plant, p| {
                const at = (p + 1) * (len / 8 + 1) + p * 3;
                if (at < len) s[at] = plant;
            }
        }
        return s;
    }

    /// Inputs for `shape` as `fill` asks, outputs poisoned and guarded; false when the kernel does not serve it.
    pub fn prepare(self: *Pair, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 512) return false;
        self.drop();
        self.rig.reset();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (self.which) {
            .add_rms => {
                if (shape.n != shape.k or shape.k == 0 or shape.k > 32768 or shape.k % 256 != 0) return false;
                const n = shape.rows * shape.k;
                self.xh = try self.part(n, false, fill, rng);
                self.ph = try self.part(n, false, fill, rng);
                self.mine = try self.rig.input(std.mem.sliceAsBytes(self.xh.?));
                self.peer = try self.rig.input(std.mem.sliceAsBytes(self.ph.?));
                const w = try self.a.alloc(u16, shape.k);
                self.wh = w;
                for (w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
                const h = try self.a.alloc(f32, n);
                self.hh = h;
                for (h) |*v| v.* = ordinary(rng);
                self.sync = try self.rig.scratch(128);
                self.w = try self.rig.input(std.mem.sliceAsBytes(w));
                self.h = try self.rig.output(n * 4);
                self.y = try self.rig.output(n * 4);
                @memcpy(self.rig.bytes(self.h, n * 4), std.mem.sliceAsBytes(h));
            },
            .stage, .gather => {
                if (shape.rows == 0 or shape.k == 0 or shape.k > 32768 or shape.k % 256 != 0) return false;
                const n = shape.rows * shape.k;
                // The gather writes this rank's half first; alternating the rank with the seed covers both orders.
                self.rank = @intCast(seed & 1);
                const mine = try self.payload(n, fill, rng);
                self.sync = try self.rig.scratch(128);
                if (self.which == .stage) {
                    self.stride = shape.k;
                    self.sh = mine;
                    self.mine = try self.rig.input(std.mem.sliceAsBytes(mine));
                    self.out = try self.rig.output(n * 2);
                } else {
                    // A frame row holds both halves, so its stride is at least 2*width: `n` sets a wider one.
                    self.stride = if (shape.n > 2 * shape.k) shape.n else 2 * shape.k;
                    self.mh = mine;
                    const peer = try self.payload(n, fill, rng);
                    self.nh = peer;
                    self.mine = try self.rig.input(std.mem.sliceAsBytes(mine));
                    self.peer = try self.rig.input(std.mem.sliceAsBytes(peer));
                    self.out = try self.rig.output(2 * ((shape.rows - 1) * self.stride + 2 * shape.k));
                }
            },
            .argmax => {
                if (shape.k == 0 or shape.k > 262144) return false;
                const n = shape.rows * shape.k;
                const k = try self.a.alloc(u16, n);
                self.kh = k;
                for (k) |*v| v.* = c.toBf16(ordinary(rng));
                if (fill == .nonfinite) {
                    const nan = c.toBf16(std.math.nan(f32));
                    const inf = c.toBf16(std.math.inf(f32));
                    const ninf = c.toBf16(-std.math.inf(f32));
                    for (0..shape.rows) |r| {
                        const row = k[r * shape.k ..][0..shape.k];
                        row[shape.k / 3] = switch (r % 4) {
                            0 => nan,
                            1 => inf,
                            2 => ninf,
                            else => row[shape.k / 3],
                        };
                        if (r + 1 == shape.rows) for (row) |*v| {
                            v.* = nan;
                        };
                    }
                }
                self.logits = try self.rig.input(std.mem.sliceAsBytes(k));
                self.out = try self.rig.output(shape.rows * 8);
            },
            .pick => {
                if (shape.n != 1 or shape.k != 1) return false;
                self.pw = try self.a.alloc(u32, 2 * shape.rows);
                self.qw = try self.a.alloc(u32, 2 * shape.rows);
                for (0..shape.rows) |r| {
                    self.pw.?[2 * r] = bit(ordinary(rng));
                    self.qw.?[2 * r] = bit(ordinary(rng));
                    self.pw.?[2 * r + 1] = rng.uintLessThan(u32, 1 << 20);
                    self.qw.?[2 * r + 1] = rng.uintLessThan(u32, 1 << 20);
                }
                if (fill == .nonfinite) {
                    const nan = bit(std.math.nan(f32));
                    const inf = bit(std.math.inf(f32));
                    const ninf = bit(-std.math.inf(f32));
                    const plant = [_]struct { m: u32, p: u32 }{
                        .{ .m = nan, .p = bit(ordinary(rng)) },
                        .{ .m = bit(ordinary(rng)), .p = inf },
                        .{ .m = ninf, .p = ninf },
                        .{ .m = nan, .p = nan },
                        .{ .m = inf, .p = nan },
                        .{ .m = bit(ordinary(rng)), .p = ninf },
                    };
                    for (plant, 0..) |x, r| if (r < shape.rows) {
                        self.pw.?[2 * r] = x.m;
                        self.qw.?[2 * r] = x.p;
                    };
                }
                self.sync = try self.rig.scratch(128);
                self.mine = try self.rig.input(std.mem.sliceAsBytes(self.pw.?));
                self.peer = try self.rig.input(std.mem.sliceAsBytes(self.qw.?));
                self.out = try self.rig.output(shape.rows * 4);
            },
        }
        self.shape = shape;
        self.rows = shape.rows;
        self.width = shape.k;
        return true;
    }

    /// One launch on the prepared buffers, waited for.
    pub fn run(self: *Pair) !void {
        const p = self.rig.begin();
        const e = p.e;
        const groups = mtl.Size.of(self.rows, 1, 1);
        const threads = mtl.Size.of(launch(self.which).threads, 1, 1);
        switch (self.which) {
            .add_rms => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                @memcpy(self.rig.bytes(self.h, self.rows * self.width * 4), std.mem.sliceAsBytes(self.hh.?));
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue(rowsArgs(self.width, self.width, 0, 1, eps), 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.peer.buf, self.peer.off, 3);
                e.setBuffer(self.h.buf, self.h.off, 4);
                e.setBuffer(self.w.buf, self.w.off, 5);
                e.setBuffer(self.y.buf, self.y.off, 6);
                e.dispatchGroups(groups, threads);
            },
            .stage => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue(rowsArgs(self.width, self.stride, self.rank, 1, 0), 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.out.buf, self.out.off, 3);
                e.dispatchGroups(groups, threads);
            },
            .gather => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                const frame = self.rig.bytes(self.out, 2 * self.count());
                @memset(frame, 0); // the frame's gaps: no launch writes them, so their planted value is zero
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue(rowsArgs(self.width, self.stride, self.rank, 1, 0), 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.peer.buf, self.peer.off, 3);
                e.setBuffer(self.out.buf, self.out.off, 4);
                e.dispatchGroups(groups, threads);
            },
            .argmax => {
                e.setPipeline(self.pipe);
                e.setBuffer(self.logits.buf, self.logits.off, 0);
                const dims = [4]u32{ @intCast(self.width), logit_lo, @intCast(self.width), 1 };
                e.setValue(dims, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                e.dispatchGroups(groups, threads);
            },
            .pick => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue([2]u32{ @intCast(self.rows), 1 }, 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.peer.buf, self.peer.off, 3);
                e.setBuffer(self.out.buf, self.out.off, 4);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), threads);
            },
        }
        try self.rig.finish(p);
    }

    /// Output elements of the prepared call: bf16 elements for the three frames, the two raw kernels' bytes.
    pub fn count(self: *Pair) usize {
        return switch (self.which) {
            .add_rms => 2 * self.rows * self.width,
            .stage => self.rows * self.width,
            .gather => (self.rows - 1) * self.stride + 2 * self.width,
            .argmax => self.rows * 8,
            .pick => self.rows * 4,
        };
    }

    /// Float64 outputs of the prepared inputs and each one's sums' term magnitude (a move's term is the value itself).
    pub fn reference(self: *Pair, want: []f64, mag: []f64) !void {
        const width = self.width;
        const total = self.rows * width;
        switch (self.which) {
            .add_rms => {
                const xh = self.xh orelse return error.NoReference;
                const ph = self.ph orelse return error.NoReference;
                const hh = self.hh orelse return error.NoReference;
                const wh = self.wh orelse return error.NoReference;
                for (0..self.rows) |r| {
                    var ss: f64 = 0;
                    for (0..width) |i| {
                        const v = @as(f64, hh[r * width + i]) + xh[r * width + i] + ph[r * width + i];
                        ss += v * v;
                    }
                    const rs = 1 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
                    for (0..width) |i| {
                        const v = @as(f64, hh[r * width + i]) + xh[r * width + i] + ph[r * width + i];
                        const w = c.fromBf16(wh[i]);
                        const at = r * width + i;
                        const span = @abs(hh[r * width + i]) + @abs(xh[r * width + i]) + @abs(ph[r * width + i]);
                        want[at] = v; // x adds three terms, so its bound's magnitude is their span, not the sum
                        mag[at] = span;
                        want[total + at] = w * v * rs; // y is that span carried through w and 1/rms
                        mag[total + at] = @abs(w) * span * rs;
                    }
                }
            },
            .stage => {
                const sh = self.sh orelse return error.NoReference;
                for (want, mag, 0..) |*w, *m, i| {
                    const v = c.fromBf16(sh[i]);
                    w.* = v;
                    m.* = @abs(v);
                }
            },
            .gather => {
                const mh = self.mh orelse return error.NoReference;
                const nh = self.nh orelse return error.NoReference;
                @memset(want, 0); // the frame's gaps: no launch writes them, so their reference value is zero
                for (0..self.rows) |row| {
                    const at = row * self.stride;
                    for (0..width) |i| {
                        want[at + i] = c.fromBf16(if (self.rank == 0) mh[row * width + i] else nh[row * width + i]);
                        want[at + width + i] = c.fromBf16(if (self.rank == 0) nh[row * width + i] else mh[row * width + i]);
                    }
                }
                for (want, mag) |w, *m| m.* = @abs(w);
            },
            else => return error.NoReference,
        }
    }

    /// The moves are exact: every output element must equal the reference's bf16 bit for bit.
    pub fn bounds(self: *Pair, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        if (self.which != .stage and self.which != .gather) return error.NotMove;
        for (allow) |*l| l.* = 0;
    }

    /// The outputs if every sum ran in `acc` in an order, the rest of the arithmetic exact (gate 4).
    pub fn simulate(self: *Pair, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.which != .add_rms) return error.NoSums;
        const width = self.width;
        const hh = self.hh orelse return error.NoSums;
        const xh = self.xh orelse return error.NoSums;
        const ph = self.ph orelse return error.NoSums;
        const wh = self.wh orelse return error.NoSums;
        const terms = try self.a.alloc(f64, width);
        defer self.a.free(terms);
        for (0..self.rows) |r| {
            for (terms, 0..) |*t, i| {
                const v = @as(f64, hh[r * width + i]) + xh[r * width + i] + ph[r * width + i];
                t.* = v * v;
            }
            const ss = rows_k.sumIn(acc, order, terms);
            const rs = 1 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
            for (0..width) |i| {
                const v = @as(f64, hh[r * width + i]) + xh[r * width + i] + ph[r * width + i];
                const at = r * width + i;
                out[at] = v;
                out[self.rows * width + at] = c.fromBf16(wh[i]) * v * rs;
            }
        }
    }

    /// The seam's own checks: the raw kernels' rule at gate 1 (non-finite inputs) and gate 3 (every served edge).
    pub fn check(self: *Pair, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        switch (self.which) {
            .argmax => {
                const k = self.kh orelse return error.NoReference;
                const n = self.width;
                for (0..self.rows) |r| {
                    var best: f32 = -std.math.inf(f32);
                    var at: u32 = 0xFFFFFFFF;
                    for (0..n) |i| {
                        const v = c.fromBf16(k[r * n + i]);
                        if (v > best) {
                            best = v;
                            at = logit_lo + @as(u32, @intCast(i));
                        }
                    }
                    const got = std.mem.readInt(u32, out[r * 8 ..][0..4], .little);
                    const id = std.mem.readInt(u32, out[r * 8 + 4 ..][0..4], .little);
                    if (got != bit(best) or id != at) return try std.fmt.bufPrint(&self.msg, "row {d}: ({e} {d}) where its own rule gives ({e} {d})", .{ r, floatOf(got), id, best, at });
                }
            },
            .pick => {
                const pw = self.pw orelse return error.NoReference;
                const qw = self.qw orelse return error.NoReference;
                for (0..self.rows) |r| {
                    const vm = floatOf(pw[2 * r]);
                    const am = pw[2 * r + 1];
                    const vp = floatOf(qw[2 * r]);
                    const ap = qw[2 * r + 1];
                    const want = if (vp > vm or (vp == vm and ap < am)) ap else am;
                    const got = std.mem.readInt(u32, out[r * 4 ..][0..4], .little);
                    if (got != want) return try std.fmt.bufPrint(&self.msg, "row {d} picks {d} where its own rule gives {d}", .{ r, got, want });
                }
            },
            else => return error.NotRaw,
        }
        return null;
    }
};
