//! Qwen3.5 BF16 scoring: full-vocabulary-normalized selected-token logprobs, a prompt's chunks fixed in size.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;

/// The most rows one scoring window keeps: a prompt's chunk here, and the verify window the engine admits.
pub const chunk_rows: usize = 128;
/// The most prompts a batch may carry, one cache each.
pub const most_requests: usize = 4;
/// The most labels a prompt may ask for: the words one block's sixteen lanes write a row.
pub const most_labels: usize = 16;

const Request = struct { input_ids: []const u32, labels: []const u32 = &.{} };
const Batch = struct { requests: []const Request };

fn writeAll(bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.c.write(1, bytes[at..].ptr, bytes.len - at);
        if (n <= 0) return error.Write;
        at += @intCast(n);
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.Usage;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, try std.fmt.parseInt(c_int, args[2], 10));
    defer context.deinit();
    const e = try E.Engine.init(gpa, io, &context, args[1], false);
    defer e.deinit();
    var work: [most_requests]E.Cache = undefined;
    var made: usize = 0;
    defer for (work[0..made]) |*cache| cache.deinit(gpa);
    for (&work) |*cache| {
        cache.* = try E.Cache.init(gpa, e, 65536);
        made += 1;
    }
    try writeAll("{\"ready\":true}\n");
    var line_: std.ArrayList(u8) = .empty;
    defer line_.deinit(gpa);
    var readbuf: [65536]u8 = undefined;
    var readat: usize = 0;
    var readlen: usize = 0;
    while (true) {
        line_.clearRetainingCapacity();
        while (true) {
            if (readat == readlen) {
                const n = std.c.read(0, &readbuf, readbuf.len);
                if (n == 0) return;
                if (n < 0) return error.Read;
                readat = 0;
                readlen = @intCast(n);
            }
            const remaining = readbuf[readat..readlen];
            const newline = std.mem.indexOfScalar(u8, remaining, '\n');
            const count = newline orelse remaining.len;
            if (line_.items.len + count > 4 * 1024 * 1024) return error.TooLarge;
            try line_.appendSlice(gpa, remaining[0..count]);
            readat += count;
            if (newline != null) {
                readat += 1;
                break;
            }
        }
        const parsed = try std.json.parseFromSlice(Batch, gpa, line_.items, .{});
        defer parsed.deinit();
        const requests = parsed.value.requests;
        if (requests.len == 0 or requests.len > most_requests) return error.InvalidBatch;
        const start = std.Io.Clock.awake.now(io).toNanoseconds();
        for (requests) |r| {
            if (r.input_ids.len == 0 or r.input_ids.len > 65530) return error.InvalidPrompt;
            for (r.input_ids) |id| if (id >= e.g.vocab) return error.InvalidToken;
            if (r.labels.len > most_labels) return error.InvalidLabelCount;
            for (r.labels) |id| if (id >= e.g.vocab) return error.InvalidLabel;
        }
        var result: std.Io.Writer.Allocating = .init(gpa);
        defer result.deinit();
        const w = &result.writer;
        try w.writeAll("{\"results\":[");
        // A prompt's forwards leave its final row in the kept window, so a prompt is scored before the next runs.
        var first = true;
        for (requests, 0..) |r, i| {
            var at: usize = 0;
            var rows: usize = 0;
            while (at < r.input_ids.len) {
                rows = @min(chunk_rows, r.input_ids.len - at);
                var parts = [_]E.Part{.{ .cache = &work[i], .rows = rows }};
                try e.forward(&parts, r.input_ids[at..][0..rows], .logits);
                at += rows;
            }
            if (!first) try w.writeByte(',');
            first = false;
            try writeScores(w, e, r.labels, rows);
        }
        const ns = std.Io.Clock.awake.now(io).toNanoseconds() - start;
        try w.print("],\"engine_ms\":{d}}}\n", .{@as(f64, @floatFromInt(ns)) / 1e6});
        try writeAll(result.written());
    }
}

/// One prompt's answer: the greedy token, then each label's logprob; the kept row is bf16, `rows - 1` is last.
fn writeScores(w: *std.Io.Writer, e: *E.Engine, labels: []const u32, rows: usize) !void {
    const last = e.rowLogits(rows - 1);
    var best: u32 = 0;
    for (last, 0..) |bits, j| {
        if (f32FromBf16(bits) > f32FromBf16(last[best])) best = @intCast(j);
    }
    // The kernel's own rule: a max shift, then a sum of exponentials, so solo and batched runs reach one result.
    var max: f32 = -std.math.inf(f32);
    for (last) |bits| max = @max(max, f32FromBf16(bits));
    var sum: f32 = 0;
    for (last) |bits| sum += @exp(f32FromBf16(bits) - max);
    const lse: f32 = max + @log(sum);
    try w.print("{{\"token\":{d},\"logprobs\":[", .{best});
    for (labels, 0..) |id, j| {
        if (j > 0) try w.writeByte(',');
        try w.print("{d}", .{@as(f64, f32FromBf16(last[id])) - @as(f64, lse)});
    }
    try w.writeAll("]}");
}

/// A kept logit's BF16 bits widened exactly, so the sum sees the bits the forward produced.
pub fn f32FromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
