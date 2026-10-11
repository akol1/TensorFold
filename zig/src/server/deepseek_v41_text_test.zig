//! Deterministic official encoder/tokenizer goldens; pinned DeepSeek-AI provenance in THIRD_PARTY_NOTICES.md.
const std = @import("std");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const HfText = @import("hf_text.zig").HfText;
const v41 = @import("deepseek_v41_text.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const reply_text = @import("reply_text.zig");
const fields = @import("fields.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const official = @embedFile("deepseek_v41_official.jinja");
const golden_bytes = @embedFile("deepseek_v41_text_goldens.json");

fn parse(a: std.mem.Allocator, bytes: []const u8) !Value {
    return (try json.parse(a, bytes)).ok;
}
fn options(c: Value) model_text.RenderOptions {
    const effort = c.field("effort");
    return .{
        .enable_thinking = c.get("thinking").?.bool,
        .reasoning_effort = if (effort) |e| (if (e == .int) e.int else e.string) else null,
        .tools = c.field("tools"),
    };
}
fn tinyCheckpoint(dir: std.Io.Dir) !void {
    const io = std.testing.io;
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0},"merges":[]},"added_tokens":[]}
    });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    try dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = official });
}

test "official V41 text reference goldens render through default checkpoint HfText" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tinyCheckpoint(tmp.dir);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    const text = try HfText.load(std.testing.allocator, std.testing.io, dir, null, a, &problem);
    defer text.deinit();
    const cases = (try parse(a, golden_bytes)).get("cases").?.array;
    for (cases) |c| {
        const rendered = text.text().render(a, c.get("messages").?, options(c), &problem) catch |err| {
            std.debug.print("V41 golden {s}: {s}\n", .{ c.strField("name").?, problem });
            return err;
        };
        try std.testing.expectEqualStrings(c.strField("prompt").?, rendered);
    }
    const user = try parse(a, "[{\"role\":\"user\",\"content\":\"hello\"}]");
    try std.testing.expectEqualStrings("<｜begin▁of▁sentence｜><｜User｜>hello", try text.text().render(a, user, .{ .add_generation_prompt = false }, &problem));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "override.jinja", .data = "override{{ messages[0].content }}" });
    const path = try std.fmt.allocPrint(a, "{s}/override.jinja", .{dir});
    const override = try HfText.load(std.testing.allocator, std.testing.io, dir, path, a, &problem);
    defer override.deinit();
    try std.testing.expectEqualStrings("overridehello", try override.text().render(a, user, .{}, &problem));
}

test "V41 real tokenizer IDs match official reference goldens when checkpoint is supplied" {
    const dir = std.testing.environ.getPosix("TENSORFOLD_V41_TOKENIZER_DIR") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const text = try HfText.load(std.testing.allocator, std.testing.io, dir, null, a, &problem);
    defer text.deinit();
    for ((try parse(a, golden_bytes)).get("cases").?.array) |c| {
        const ids = try text.text().renderIds(a, c.get("messages").?, options(c), &problem);
        const expected = c.get("ids").?.array;
        const wanted = try a.alloc(u32, expected.len);
        for (expected, wanted) |id, *slot| slot.* = @intCast(id.int64().?);
        try std.testing.expectEqualSlices(u32, wanted, ids);
    }
}

test "V41 numeric reasoning efforts are accepted without changing other templates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const levels = try fields.effortLevels(a, official);
    const effort = try fields.thinkingFields(&cx, try parse(a, "{\"reasoning_effort\":42}"), levels);
    try std.testing.expectEqualStrings("42", effort.effort.?);
    try std.testing.expect(effort.enable.?);
    for ([_][]const u8{ "0", "101", "true", "1.5" }) |value| {
        const body = try parse(a, try std.mem.concat(a, u8, &.{ "{\"reasoning_effort\":", value, "}" }));
        try std.testing.expectError(error.Refused, fields.thinkingFields(&cx, body, levels));
    }
    try std.testing.expectError(error.Refused, fields.thinkingFields(&cx, try parse(a, "{\"reasoning_effort\":42}"), &.{ "low", "high" }));
    const numeric = try v41.effort(a, official, "42");
    try std.testing.expectEqual(@as(i64, 42), numeric.integer);
    try std.testing.expectEqualStrings("max", (try v41.effort(a, official, "xhigh")).string);
    try std.testing.expectError(error.Template, v41.effort(a, official, "0"));
    try std.testing.expectError(error.Template, v41.effort(a, official, "101"));
}

test "V41 text preparation rejects media at every supported nesting boundary" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    for ([_][]const u8{
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"image.png\"}}]}]",
        "[{\"role\":\"tool\",\"content\":[{\"type\":\"image\",\"source\":{\"data\":\"x\"}}]}]",
        "[{\"role\":\"user\",\"content\":\"<image>image.png</image>\"}]",
        "[{\"role\":\"user\",\"content\":\"<｜deepseek_image｜>\"}]",
        "[{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"<｜deepseek_image｜>\"}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"x\"}}]}]",
        "[{\"role\":\"user\",\"content_blocks\":[{\"type\":\"video_url\",\"video_url\":\"clip.mp4\"}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"content\":[{\"type\":\"image\",\"source\":\"x\"}]}]}]",
        "[{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"<image>image.png</image>\"}]",
    }) |input| try std.testing.expectError(error.Template, v41.prepare(a, try parse(a, input), &problem));
}

test "V41 image-named tool arguments remain ordinary typed tool data" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tinyCheckpoint(tmp.dir);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    const text = try HfText.load(std.testing.allocator, std.testing.io, dir, null, a, &problem);
    defer text.deinit();
    const messages = try parse(a, "[{\"role\":\"user\",\"content\":\"inspect it\"}," ++
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"inspect\"," ++
        "\"arguments\":{\"image\":\"cat.jpg\",\"audio\":true}}}]}," ++
        "{\"role\":\"tool\",\"content\":\"done\"},{\"role\":\"user\",\"content\":\"continue\"}]");
    const rendered = try text.text().render(a, messages, .{ .enable_thinking = false }, &problem);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "<｜DSML｜ parameter name=\"image\" string=\"true\">cat.jpg</｜DSML｜ parameter>") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "<｜DSML｜ parameter name=\"audio\" string=\"false\">true</｜DSML｜ parameter>") != null);
}

const completion = "summary\n\n<｜DSML｜ calls>\n" ++
    "<｜DSML｜ invoke name=\"search::lookup\">\n" ++
    "<｜DSML｜ parameter name=\"query\" string=\"true\">  value\n\"quoted\"  </｜DSML｜ parameter>\n" ++
    "<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n" ++
    "<｜DSML｜ parameter name=\"data\" string=\"false\">[true,null,{\"x\":1}]</｜DSML｜ parameter>\n" ++
    "</｜DSML｜ invoke>\n" ++
    "<｜DSML｜ invoke name=\"files::lookup\">\n</｜DSML｜ invoke>\n" ++
    "</｜DSML｜ calls>";
const offered = "[{\"type\":\"function\",\"namespace\":\"search\",\"function\":{\"name\":\"lookup\"}}," ++
    "{\"type\":\"function\",\"namespace\":{\"name\":\"files\"},\"function\":{\"name\":\"lookup\"}}]";

test "V41 leading-space DSML preserves namespace and typed parameter boundaries" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try parse(a, offered)).array;
    const parsed = try tool_parse.parse(a, completion, tools, null);
    try std.testing.expectEqualStrings("summary", parsed.content);
    try std.testing.expectEqual(@as(usize, 2), parsed.calls.?.len);
    const first = parsed.calls.?[0];
    try std.testing.expectEqualStrings("search", first.strField("namespace").?);
    try std.testing.expectEqualStrings("lookup", first.get("function").?.strField("name").?);
    const arguments = try parse(a, first.get("function").?.strField("arguments").?);
    try std.testing.expectEqualStrings("  value\n\"quoted\"  ", arguments.strField("query").?);
    try std.testing.expectEqual(@as(i64, 2), arguments.get("limit").?.int64().?);
    try std.testing.expect(json.equal(arguments.get("data").?, try parse(a, "[true,null,{\"x\":1}]")));
    try std.testing.expectEqualStrings("files", parsed.calls.?[1].strField("namespace").?);
    try std.testing.expectEqualStrings("{}", parsed.calls.?[1].get("function").?.strField("arguments").?);
    const old = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"q\" string=\"true\">v</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
    const old_tools = (try parse(a, "[{\"function\":{\"name\":\"lookup\"}}]")).array;
    try std.testing.expect((try tool_parse.parse(a, old, old_tools, null)).calls != null);
}

test "V41 DSML streaming survives every byte boundary and emits each validated call once" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try parse(a, offered)).array;
    var streamer = try tool_stream.Streamer.init(a, tools);
    var deltas: std.ArrayList(Value) = .empty;
    for (0..completion.len + 1) |end| try streamer.feed(completion[0..end], &deltas);
    try streamer.feed(completion, &deltas);
    try std.testing.expect(streamer.streamed and !streamer.invalidDsml(completion));
    try std.testing.expectEqual(@as(usize, 4), deltas.items.len);
    try std.testing.expectEqualStrings("search", deltas.items[0].get("tool_calls").?.array[0].strField("namespace").?);
    try std.testing.expectEqual(@as(i64, 1), deltas.items[2].get("tool_calls").?.array[0].get("index").?.int64().?);
    try std.testing.expectEqualStrings("files", deltas.items[2].get("tool_calls").?.array[0].strField("namespace").?);
    try std.testing.expectEqualStrings("summary\n\n", try reply_text.hideToolCalls(a, completion, true));
}

test "V41 malformed and incomplete DSML never silently truncates into a valid call" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try parse(a, offered)).array;
    for ([_][]const u8{
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search::lookup\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">bad json</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search::lookup\">\n<｜DSML｜ parameter name=\"x\" string=\"true\">a</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"x\" string=\"true\">b</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search::lookup\">\n",
    }) |bad| {
        const parsed = try tool_parse.parse(a, bad, tools, 1);
        try std.testing.expect(parsed.calls == null);
        try std.testing.expectEqualStrings(reply_text.pyStrip(bad), reply_text.pyStrip(parsed.content));
        var streamer = try tool_stream.Streamer.init(a, tools);
        var deltas: std.ArrayList(Value) = .empty;
        try streamer.feed(bad, &deltas);
        try std.testing.expectEqual(@as(usize, 0), deltas.items.len);
        try std.testing.expect(streamer.invalidDsml(bad));
    }
}

test "V41 completion parsing matches independently generated official decode goldens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try parse(a, "[" ++
        "{\"function\":{\"name\":\"lookup\"}}," ++
        "{\"namespace\":\"search\",\"function\":{\"name\":\"lookup\"}}," ++
        "{\"namespace\":\"files\",\"function\":{\"name\":\"lookup\"}}]")).array;
    const goldens = try parse(a, @embedFile("deepseek_v41_tool_goldens.json"));
    for (goldens.get("cases").?.array) |c| {
        const raw = c.strField("completion").?;
        const eos = "<｜end▁of▁sentence｜>";
        const complete = raw[0 .. raw.len - eos.len];
        const expected = c.get("parsed").?;
        const split = if (c.get("thinking").?.bool)
            try reply_text.splitThinking(a, complete, true, reply_text.think_markers)
        else
            reply_text.Split{ .reasoning = "", .answer = complete };
        try std.testing.expectEqualStrings(expected.strField("reasoning_content").?, split.reasoning);
        const parsed = try tool_parse.parse(a, split.answer, tools, null);
        try std.testing.expectEqualStrings(expected.strField("content").?, parsed.content);
        const wanted = expected.get("tool_calls").?.array;
        try std.testing.expectEqual(wanted.len, parsed.calls.?.len);
        for (wanted, parsed.calls.?) |reference_call, actual| {
            try std.testing.expectEqualStrings(json.strOr(reference_call.get("namespace")), json.strOr(actual.get("namespace")));
            const ref_fn = reference_call.get("function").?;
            const actual_fn = actual.get("function").?;
            try std.testing.expectEqualStrings(ref_fn.strField("name").?, actual_fn.strField("name").?);
            try std.testing.expect(json.equal(try parse(a, ref_fn.strField("arguments").?), try parse(a, actual_fn.strField("arguments").?)));
        }
    }
}
