//! The lane backend's window timing: each width's ms on a scratch slot 0, kept in the per-build cost cache.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;

const nl = @import("qwen27_lanes.zig"); // the native backend, not the core lane engine `lanes` names
const Metal = nl.Metal;
const max_rows = nl.max_rows;

/// Each window width's ms (verify, draws, keep) on a scratch context in slot 0; resets it. Kept per build.
pub fn measure(b: *Metal, io: std.Io) !void {
    const r = b.runner;
    if (b.owners[0] != null or r.active != null) return error.StreamBusy;
    var shape: [128]u8 = undefined;
    const parts = [_][]const u8{ "qwen27-lanes", std.mem.span(r.model.device.name()), try q.widths.costShape(r.model.config, r.model.widths_tag, &shape, max_rows, b.draft != null) };
    const k = lanes.cost_cache.key(b.gpa, io, &parts) catch null;
    if (k) |key| if (lanes.cost_cache.load([max_rows]lanes.config.Cost, b.gpa, io, key)) |kept| {
        b.costs = kept;
        b.timed = max_rows;
        return;
    };
    if (r.capacity <= max_rows) return error.ContextFull;
    defer {
        if (b.draft) |d| d.reset() catch {} else r.reset(0) catch {};
    }
    var ids: [192]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
    const prompt = ids[0..@min(ids.len, r.capacity - max_rows)]; // a small --context still holds a window past it
    try b.refill(prompt);
    var s = try lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = prompt, .max_new = 1 });
    defer s.deinit(b.gpa);
    const timer: WindowTimer = .{ .b = b, .s = &s, .prompt = prompt };
    for (0..5) |_| _ = try timer.time(max_rows - 1); // the widest window first, while the GPU's clocks ramp up
    var ms: [max_rows]f64 = undefined;
    for (&ms, 0..) |*m, i| m.* = try lanes.cost_rule.fastest(timer, i);
    try lanes.cost_rule.smooth(timer, &ms);
    const ref = lanes.cost_cache.referenceKey(&parts);
    if (lanes.cost_cache.load([max_rows]f64, b.gpa, io, ref)) |reference| if (lanes.cost_rule.drifted(&ms, &reference)) try lanes.cost_rule.again(timer, &ms);
    for (&b.costs, ms, 0..) |*cost, m, i| cost.* = .{ .width = @intCast(i + 1), .ms = m };
    b.timed = max_rows;
    if (k) |key| lanes.cost_cache.save([max_rows]lanes.config.Cost, b.gpa, io, key, b.costs);
    lanes.cost_cache.save([max_rows]f64, b.gpa, io, ref, ms);
}

/// Window entry `i` is `i + 1` rows, timed once; each keeps a row, so a full slot is refilled first.
const WindowTimer = struct {
    b: *Metal,
    s: *lanes.Stream,
    prompt: []const u32,
    pub fn time(t: WindowTimer, i: usize) !f64 {
        if (t.b.runner.offsets[0] + i + 1 > t.b.runner.capacity) try t.b.refill(t.prompt);
        return timeWindow(t.b, t.s, i + 1);
    }
};

pub fn timeWindow(b: *Metal, s: *lanes.Stream, rows: usize) !f64 {
    const t0 = mtl.clock.seconds();
    const positions: [max_rows]u64 = @splat(0);
    const drafts: [max_rows]u32 = @splat(1);
    var sampled: [max_rows]u32 = undefined;
    var echoed: [max_rows]u32 = undefined;
    b.owners[0] = s;
    defer b.owners[0] = null;
    const w = be.Window{ .stream = s, .pending = 0, .held = 0, .tokens = drafts[0 .. rows - 1], .parents = null, .positions = positions[0..rows] };
    var out = [_]be.Verified{.{ .sampled = sampled[0..rows], .drafts = echoed[0 .. rows - 1] }};
    try nl.Metal.verifyFn(b, &.{w}, &out);
    var path = [_]u32{0};
    try b.keepPaths(&.{&path});
    return (mtl.clock.seconds() - t0) * 1e3;
}
