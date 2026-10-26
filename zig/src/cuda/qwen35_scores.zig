//! qwen35_scores.cu: a row's full-vocabulary log-sum-exp, its argmax and the selected labels' logits.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");

pub const threads = 256;
/// A row's stats: log-sum-exp and argmax at `rows * 2` words, each row's 16 selected logits after `8`.
pub const stats_base = 8;
pub const stats_width = 16;
pub const labels_width = 17;

pub const Kernels = struct {
    mod: Module,
    stats_fn: Function,

    pub fn resolve(mod: Module) !Kernels {
        return .{ .mod = mod, .stats_fn = try mod.function("tf_q35_score_stats") };
    }

    /// Load the embedded module (owned: call unload) and resolve its functions.
    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qwen35_scores);
        errdefer mod.unload();
        return resolve(mod);
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// `stats` words a run of `rows` rows writes: two a row, then sixteen a row from `stats_base`.
    pub fn statsWords(rows: usize) usize {
        return stats_base + rows * stats_width;
    }

    /// A row's log-sum-exp, argmax and selected logits, one block a row, at most 128 rows, batch independent.
    pub fn stats(k: *const Kernels, s: Stream, logits: u64, vocab: usize, out: u64, labels: u64, rows: usize) !void {
        if (rows == 0) return;
        if (vocab == 0 or vocab > std.math.maxInt(i32) or rows > 128) return error.RowShape;
        var a: launch.Args = .{};
        a.add(logits);
        a.add(@as(i32, @intCast(vocab)));
        a.add(out);
        a.add(labels);
        try launch.launch(k.stats_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};

test "score stats: two words a row stay clear of the selected logits that start at eight" {
    // A row writes its log-sum-exp and argmax at 2r, and its sixteen logits at 8 + 16r: the halves never overlap.
    for ([_]usize{ 1, 2, 4 }) |rows| {
        try std.testing.expect(rows * 2 <= stats_base);
        try std.testing.expectEqual(stats_base + rows * stats_width, Kernels.statsWords(rows));
    }
    // A label list is a count then its ids, one word a row beyond the sixteen a row the kernel reads.
    try std.testing.expectEqual(@as(usize, 17), labels_width);
    try std.testing.expect(labels_width == 1 + stats_width);
    // One block a row is what keeps a row's numbers independent of the launch's other rows.
    try std.testing.expectEqual(@as(u32, 256), threads);
}
