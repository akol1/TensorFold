//! Real text-interface decoder failures must survive through the consumer error boundary.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;

fn failedDecode(_: *anyopaque, _: std.mem.Allocator, _: []const u32) model_text.Error![]u8 {
    return error.Template;
}

fn failedBytes(_: *anyopaque, _: std.mem.Allocator, _: u32) model_text.Error!?[]u8 {
    return error.Template;
}

fn info(_: *anyopaque) api.Info {
    return .{ .context_window = 4096, .logprobs = true };
}

const Output = struct {
    opened: bool = false,
    status: ?u16 = null,
    decoder_error: bool = false,
    content: bool = false,

    fn out(o: *@This()) openai.Out {
        return .{ .ctx = o, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque) error{Closed}!void {
        const o: *Output = @ptrCast(@alignCast(ctx));
        o.opened = true;
    }
    fn observe(o: *Output, payload: json.Value) void {
        if (payload.get("error")) |err| if (err.strField("message")) |message| {
            o.decoder_error = std.mem.indexOf(u8, message, "tokenizer") != null and
                std.mem.indexOf(u8, message, "decode") != null and std.mem.indexOf(u8, message, "memory") == null;
        };
        if (payload.get("choices")) |choices| if (choices == .array) for (choices.array) |choice| {
            const message = choice.get("delta") orelse choice.get("message") orelse continue;
            if (message.strField("content")) |content| o.content = o.content or content.len > 0;
        };
    }
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const o: *Output = @ptrCast(@alignCast(ctx));
        if (payload) |p| o.observe(p);
    }
    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const o: *Output = @ptrCast(@alignCast(ctx));
        o.status = status;
        o.observe(payload);
    }
};

pub fn run(comptime TestText: type, comptime HiEngine: type) !void {
    const ReplyDecoder = struct {
        fn decode(ctx: *anyopaque, a: std.mem.Allocator, ids: []const u32) model_text.Error![]u8 {
            if (ids.len > 0 and ids[0] == 'h') return error.Template;
            const original: *TestText = @ptrCast(@alignCast(ctx));
            return original.text().decode(a, ids);
        }
    };
    for (0..3) |mode| for ([_]bool{ false, true }) |stream| {
        var baseline: TestText = .{};
        const base = baseline.text();
        var text_vtable = base.vtable.*;
        switch (mode) {
            0 => text_vtable.decode = failedDecode,
            1 => text_vtable.token_bytes = failedBytes,
            else => text_vtable.decode = ReplyDecoder.decode,
        }
        var backend: HiEngine = .{};
        const engine = backend.engine();
        var engine_vtable = engine.vtable.*;
        engine_vtable.info = info;
        const srv = try Server.init(std.testing.allocator, std.testing.io, .{ .ctx = engine.ctx, .vtable = &engine_vtable }, .{ .ctx = base.ctx, .vtable = &text_vtable }, .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .enable_thinking = false, .use_drafts = false }, null);
        defer srv.deinit();
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const fields = switch (mode) {
            0 => "\"prompt\":[[120]]",
            1 => "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"logprobs\":true",
            else => "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]",
        };
        const raw = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":2,{s},\"stream\":{s}}}", .{ fields, if (stream) "true" else "false" });
        var seen: Output = .{};
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        openai.run(srv, a, seen.out(), .{ .conn = &conn }, mode != 0, (try json.parse(a, raw)).ok);
        try std.testing.expect(seen.decoder_error);
        try std.testing.expect(!seen.content);
        if (mode < 2 or !stream) {
            try std.testing.expectEqual(@as(?u16, 500), seen.status);
            try std.testing.expect(!seen.opened);
        } else {
            try std.testing.expect(seen.opened);
            try std.testing.expectEqual(@as(?u16, null), seen.status);
        }
        try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
    };
}
