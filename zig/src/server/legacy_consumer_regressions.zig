//! Containment final-review consumer regressions, exercised by the preflight suite.
const std = @import("std");
const json = @import("json.zig");

const dsml_prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
    "<｜DSML｜parameter name=\"q\" string=\"true\">";
const dsml_suffix = "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
const dsml_body = "{\"model\":\"test-model\",\"max_tokens\":2000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";

pub fn run(comptime TestText: type, comptime CallEngine: type, comptime sendWith: anytype) !void {
    // DSML literal calls and adjacent XML calls agree in whole and streamed consumer replies.
    {
        const literals = [_][]const u8{
            "<tool_call><function=lookup></function></tool_call>",
            "<tool_call><function=lookup></function>",
        };
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const adjacent = "<tool_call><function=lookup><parameter=n>7</parameter></function></tool_call>";
        for (literals) |literal| for ([_]bool{ false, true }) |with_adjacent| {
            const raw = try std.mem.concat(a, u8, &.{ dsml_prefix, literal, dsml_suffix, if (with_adjacent) adjacent else "" });
            const tokens = try TestText.encode(undefined, a, raw, false);
            for ([_]bool{ false, true }) |single| {
                const expected = try std.mem.concat(a, u8, &.{
                    "{\"q\":", try json.quote(a, literal, .{}), "}", if (with_adjacent and !single) "{\"n\":\"7\"}" else "",
                });
                const body = try std.mem.concat(a, u8, &.{ dsml_body, if (single) ",\"parallel_tool_calls\":false" else "" });
                for ([_]bool{ false, true }) |batched| {
                    var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                    const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
                    const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
                    try std.testing.expectEqual(@as(?u16, 200), whole.status);
                    try std.testing.expect(streamed.opened and streamed.status == null);
                    try std.testing.expectEqual(@as(usize, if (with_adjacent and !single) 2 else 1), whole.calls);
                    try std.testing.expectEqual(whole.calls, streamed.calls);
                    try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
                    try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
                    try std.testing.expectEqualStrings("", whole.content());
                    try std.testing.expectEqualStrings("", streamed.content());
                    try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
                }
            }
        };
    }
    // unfinished and malformed DSML never emit their complete nested call in consumer replies.
    {
        const prefix = dsml_prefix ++ "<tool_call><function=lookup></function></tool_call>";
        const replies = [_][]const u8{
            prefix,
            prefix ++ "</｜DSML｜parameter></｜DSML｜tool_calls>",
        };
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        for (replies) |raw| {
            const tokens = try TestText.encode(undefined, a, raw, false);
            for ([_]bool{ false, true }) |single| for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const body = try std.mem.concat(a, u8, &.{ dsml_body, if (single) ",\"parallel_tool_calls\":false" else "" });
                const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
                const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 0), whole.calls);
                try std.testing.expectEqual(@as(usize, 0), streamed.calls);
                try std.testing.expectEqual(@as(usize, 0), streamed.arguments_len);
                try std.testing.expectEqualStrings(raw, whole.content());
                try std.testing.expectEqualStrings(raw, streamed.content());
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            };
        }
    }
    // DSML and adjacent existing non-XML envelopes preserve whole and SSE calls without duplicates.
    {
        const outer = dsml_prefix ++ "outer" ++ dsml_suffix;
        const cases = [_][2][]const u8{
            .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"adjacent-json\"}}</tool_call>", "{\"q\":\"adjacent-json\"}" },
            .{ "<tool_call>lookup<arg_key>q</arg_key><arg_value>adjacent-glm</arg_value></tool_call>", "{\"q\":\"adjacent-glm\"}" },
            .{ "<|tool_call>call:lookup{q:<|\"|>adjacent-gemma<|\"|>}<tool_call|>", "{\"q\":\"adjacent-gemma\"}" },
            .{ "<tools:tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"adjacent-namespaced\"}}</tools:tool_call>", "{\"q\":\"adjacent-namespaced\"}" },
        };
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        for (cases) |case| for ([_]bool{ false, true }) |outer_first| {
            const raw = try std.mem.concat(a, u8, if (outer_first) &.{ outer, case[0] } else &.{ case[0], outer });
            const expected = try std.mem.concat(a, u8, if (outer_first)
                &.{ "{\"q\":\"outer\"}", case[1] }
            else
                &.{ case[1], "{\"q\":\"outer\"}" });
            const tokens = try TestText.encode(undefined, a, raw, false);
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
                const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 2), whole.calls);
                try std.testing.expectEqual(@as(usize, 2), streamed.calls);
                try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
                try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        };
    }
    // a supported bare JSON call containing a literal DSML opener agrees in whole and SSE replies.
    {
        const call = "{\"name\":\"lookup\",\"arguments\":{\"q\":\"<｜DSML｜tool_calls>\"}}";
        const replies = [_][]const u8{ call, "[" ++ call ++ "]", "```json\n" ++ call ++ "\n```" };
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        for (replies) |raw| {
            const tokens = try TestText.encode(undefined, a, raw, false);
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
                const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 1), whole.calls);
                try std.testing.expectEqual(@as(usize, 1), streamed.calls);
                const expected = "{\"q\":\"<｜DSML｜tool_calls>\"}";
                try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
                try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        }
    }
    // closed rejected calls and literal less-than prose preserve later consumer calls.
    {
        const outer = dsml_prefix ++ "outer" ++ dsml_suffix;
        const adjacent = "<tool_call><function=lookup><parameter=q>adjacent</parameter></function></tool_call>";
        const cases = [_][2][]const u8{
            .{ outer ++ "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>" ++ adjacent, "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ outer ++ "<tool_call><function=other></function></tool_call>" ++ adjacent, "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ outer ++ "<tool_call></tool_call>" ++ adjacent, "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ outer ++ "<tool_call><not_a_function></not_a_function></tool_call>" ++ adjacent, "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ adjacent ++ "1 < 2 " ++ adjacent, "{\"q\":\"adjacent\"}{\"q\":\"adjacent\"}" },
            .{ "1 < 2 " ++ outer ++ adjacent, "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
        };
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        for (cases) |case| {
            const tokens = try TestText.encode(undefined, a, case[0], false);
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
                const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 2), whole.calls);
                try std.testing.expectEqual(@as(usize, 2), streamed.calls);
                try std.testing.expectEqualStrings(case[1], whole.arguments[0..whole.arguments_len]);
                try std.testing.expectEqualStrings(case[1], streamed.arguments[0..streamed.arguments_len]);
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        }
    }
}

pub fn mixedChain(comptime TestText: type, comptime CallEngine: type, comptime sendWith: anytype) !void {
    const xml = "<tool_call><function=lookup><parameter=q>xml</parameter></function></tool_call>";
    const v41 = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n" ++
        "<｜DSML｜ parameter name=\"q\" string=\"true\">v41</｜DSML｜ parameter>\n" ++
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const cases = [_][]const u8{
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"middle\"}}</tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>middle</arg_value></tool_call>",
        "<|tool_call>call:lookup{q:<|\"|>middle<|\"|>}<tool_call|>",
        "<tools:tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"middle\"}}</tools:tool_call>",
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |middle| {
        const raw = try std.mem.concat(a, u8, &.{ xml, middle, v41, xml });
        const tokens = try TestText.encode(undefined, a, raw, false);
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
            const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
            try std.testing.expectEqual(@as(?u16, 200), whole.status);
            try std.testing.expect(streamed.opened and streamed.status == null);
            try std.testing.expect(!whole.got_error and !streamed.got_error);
            try std.testing.expectEqual(@as(usize, 4), whole.calls);
            try std.testing.expectEqual(whole.calls, streamed.calls);
            const expected = "{\"q\":\"xml\"}{\"q\":\"middle\"}{\"q\":\"v41\"}{\"q\":\"xml\"}";
            try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
            try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
            try std.testing.expectEqualStrings(whole.content(), streamed.content());
            try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
        }
    }
}
