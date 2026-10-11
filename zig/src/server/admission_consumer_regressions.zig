//! Consumer-visible refusals that must precede opening the event stream.
const std = @import("std");
const api = @import("engine_api");

pub fn budgetWithoutThinking(engine: api.Engine, comptime sendWith: anytype) !void {
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"max_tokens\":2,\"temperature\":0,\"thinking_budget\":5," ++
        "\"chat_template_kwargs\":{\"enable_thinking\":false}";
    for ([_][]const u8{ ",\"stream\":false}", ",\"stream\":true}" }) |suffix| {
        const raw = try std.mem.concat(std.testing.allocator, u8, &.{ body, suffix });
        defer std.testing.allocator.free(raw);
        const seen = try sendWith(engine, .chat, raw, .{});
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(!seen.opened);
        try std.testing.expectEqual(@as(i64, 0), seen.preparing);
    }
}

pub fn media(comptime send: anytype) !void {
    const bodies = [_][]const u8{
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":[" ++
            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/image.png\"}}]}],",
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
            "\"modalities\":[\"text\",\"audio\"],",
    };
    for (bodies) |body| for ([_][]const u8{ "\"stream\":false}", "\"stream\":true}" }) |suffix| {
        const raw = try std.mem.concat(std.testing.allocator, u8, &.{ body, suffix });
        defer std.testing.allocator.free(raw);
        const seen = try send(.chat, raw);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(!seen.opened);
        try std.testing.expectEqual(@as(i64, 0), seen.preparing);
    };
}
