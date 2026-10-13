//! Family stage labels over the generic same-command-buffer counter recorder; labels are CPU-side
//! bookkeeping only and never add a GPU readback, wait or allocation while profiling is disabled.
const std = @import("std");
const mtl = @import("metal");
const counters = @import("../../core/gpu_counters.zig");

pub const Stage = enum { engram, mhc, attention, moe, head, other };
pub const stage_count = @typeInfo(Stage).@"enum".field_names.len;
pub const Options = counters.Options;
pub const Aggregate = counters.Aggregate;
pub const Span = counters.Span;

pub fn class(stage: Stage) counters.Class {
    return switch (stage) {
        .engram => .embedding,
        .mhc => .norm,
        .attention => .attention,
        .moe, .head => .dense,
        .other => .copy,
    };
}

/// Unique owner of the recorder; span labels and class totals share the recorder's bounded span storage.
pub const Recorder = struct {
    recorder: counters.Recorder,
    allocator: std.mem.Allocator,
    labels: []Stage = &.{},
    totals: [stage_count]Aggregate = @splat(.{}),

    pub fn init(allocator: std.mem.Allocator, device: mtl.Device, options: Options) !Recorder {
        var self = Recorder{
            .recorder = try counters.Recorder.init(allocator, device, options),
            .allocator = allocator,
        };
        errdefer self.recorder.deinit() catch {};
        if (self.recorder.enabled()) self.labels = try allocator.alloc(Stage, options.max_spans);
        return self;
    }
    pub fn enabled(self: *const Recorder) bool {
        return self.recorder.enabled();
    }
    pub fn capability(self: *const Recorder) mtl.counters.Capability {
        return self.recorder.capability();
    }
    pub fn activeSpan(self: *const Recorder) bool {
        return self.recorder.active_span != null;
    }
    pub fn beginCommand(self: *Recorder, cb: mtl.CommandBuffer) !void {
        try self.recorder.beginCommand(cb);
        self.totals = @splat(.{});
    }
    pub fn beginSpan(self: *Recorder, encoder: *mtl.ComputeEncoder, stage: Stage) !void {
        try self.recorder.beginSpan(encoder, class(stage));
        if (self.recorder.enabled()) self.labels[self.recorder.span_count - 1] = stage;
    }
    /// Stage of the span at `index` in the recorder's span order; only reading after completion is valid.
    pub fn stageOf(self: *const Recorder, index: usize) !Stage {
        if (index >= self.recorder.span_count) return error.InvalidProfileStage;
        return self.labels[index];
    }
    pub fn endSpan(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        try self.recorder.endSpan(encoder);
    }
    pub fn finishEncoding(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        try self.recorder.finishEncoding(encoder);
    }
    pub fn cancelEncoding(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        try self.recorder.cancelEncoding(encoder);
        self.totals = @splat(.{});
    }
    pub fn discardAfterCompletion(self: *Recorder) !void {
        try self.recorder.discardAfterCompletion();
        self.totals = @splat(.{});
    }
    pub fn readAfterCompletion(self: *Recorder) ![]const Span {
        const spans = try self.recorder.readAfterCompletion();
        var totals: [stage_count]Aggregate = @splat(.{});
        for (spans, 0..) |span, i| {
            const total = &totals[@backingInt(self.labels[i])];
            total.calls += 1;
            total.gpu_seconds += span.gpu_seconds;
        }
        self.totals = totals;
        return spans;
    }
    pub fn aggregates(self: *const Recorder) [stage_count]Aggregate {
        return self.totals;
    }
    pub fn deinit(self: *Recorder) !void {
        const labels = self.labels;
        self.labels = &.{};
        defer if (labels.len != 0) self.allocator.free(labels);
        try self.recorder.deinit();
        self.totals = @splat(.{});
    }
};

test "disabled family profiling allocates nothing and leaves no stage receipts" {
    var recorder = try Recorder.init(std.testing.failing_allocator, undefined, .{});
    defer recorder.deinit() catch unreachable;
    var encoder: mtl.ComputeEncoder = undefined;
    try recorder.beginSpan(&encoder, .mhc);
    try recorder.endSpan(&encoder);
    try std.testing.expect(!recorder.activeSpan());
    try std.testing.expectEqual(@as(usize, 0), (try recorder.readAfterCompletion()).len);
    for (recorder.aggregates()) |total| try std.testing.expectEqual(@as(u64, 0), total.calls);
}
