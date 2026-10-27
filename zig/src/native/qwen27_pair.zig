//! The pair transport's types, the seam ABI the engine calls, and the window one exchange may carry.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;

/// The pair's transport type, named through glm_dsa so this file compiles in both module graphs (see core/pair.zig).
pub const Pair = std.meta.Child(@FieldType(tf.glm_dsa.engine.Engine, "pair"));

/// The command set rank 1 replays: the four GPU operations a pair round is made of (glm_dsa's `follow.zig` shape).
pub const Kind = enum(u32) { begin = 1, chunk, step, end };

/// Rank 0 puts each round on the wire before running it; rank 1 replays them in order here. Owned by the host.
pub const Link = struct {
    pair: *Pair,
    gpa: std.mem.Allocator,
    words: std.ArrayList(u32) = .empty,

    pub fn deinit(l: *Link) void {
        l.words.deinit(l.gpa);
    }

    fn put(l: *Link, kind: Kind) !void {
        try l.pair.send(@backingInt(kind), l.words.items);
        l.words.clearRetainingCapacity();
    }

    /// A prompt's chunk boundaries are rank 0's memory policy's, so the send carries them: rank 1 cannot infer them.
    pub fn begin(l: *Link, slot: u32) !void {
        l.words.clearRetainingCapacity();
        try l.words.append(l.gpa, slot);
        try l.put(.begin);
    }

    pub fn chunk(l: *Link, slot: u32, ids: []const u32, last: bool, decoded: bool) !void {
        l.words.clearRetainingCapacity();
        try l.words.appendSlice(l.gpa, &.{ slot, @intCast(ids.len), @intFromBool(last), @intFromBool(decoded) });
        try l.words.appendSlice(l.gpa, ids);
        try l.put(.chunk);
    }

    pub fn step(l: *Link, slot: u32, token: u32) !void {
        l.words.clearRetainingCapacity();
        try l.words.appendSlice(l.gpa, &.{ slot, token });
        try l.put(.step);
    }

    pub fn end(l: *Link, slot: u32) void {
        l.words.clearRetainingCapacity();
        l.words.append(l.gpa, slot) catch return;
        l.put(.end) catch {};
    }
};

/// The pair's seam ends in the engine's shape: the transport's `Ref` is the frame's, so the casts are field copies.
pub fn pairReduce(p: *Pair) q.forward.Reduce {
    return .{ .ptr = p, .part = pairPart, .stage = pairStage, .gather = pairGather, .failed = pairFailed, .argmax = pairArgmax };
}

fn pairPart(ptr: *anyopaque) q.forward.Seam {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    const s = p.part();
    return .{ .x = s.x, .ref = .{ .buffer = s.ref.buf, .offset = s.ref.off } };
}

fn pairStage(ptr: *anyopaque, e: *mtl.ComputeEncoder, x: u32, src: q.forward.Ref, rows: u32, width: u32, stride: u32) void {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    p.stage(e, x, .{ .buf = src.buffer, .off = src.offset }, rows, width, stride);
}

fn pairGather(ptr: *anyopaque, e: *mtl.ComputeEncoder, x: u32, mine: q.forward.Ref, out: q.forward.Ref, rows: u32, width: u32, out_stride: u32) void {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    p.gather(e, x, .{ .buf = mine.buffer, .off = mine.offset }, .{ .buf = out.buffer, .off = out.offset }, rows, width, out_stride);
}

fn pairFailed(ptr: *anyopaque) bool {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    return p.failed();
}

fn pairArgmax(ptr: *anyopaque, e: *mtl.ComputeEncoder, x: u32, logits: q.forward.Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: q.forward.Ref) void {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    p.argmaxBf16(e, x, .{ .buf = logits.buffer, .off = logits.offset }, stride, lo, n, rows, .{ .buf = picks.buffer, .off = picks.offset });
}

/// The window an exchange may carry: a round's rows of the widest seam payload -- this rank's `intermediate / 2` bf16.
pub fn pairBytes() usize {
    return @as(usize, batch_rows) * @divExact(17408, 2) * @sizeOf(u16);
}

/// A shared round's rows at most: the frame's.
pub const batch_rows = 128;
