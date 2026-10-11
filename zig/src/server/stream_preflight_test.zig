//! A streamed context-window refusal is an error event, and every other refusal is a 400 before the stream opens.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic_translate = @import("anthropic_translate.zig");
const responses_translate = @import("responses_translate.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;

/// Unicode scalars as tokens; prompts are six tokens unless the official V41 renderer is selected.
const TestText = struct {
    v41: bool = false,
    pub fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    pub fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const view = std.unicode.Utf8View.init(input) catch return error.Template;
        const count = std.unicode.utf8CountCodepoints(input) catch return error.Template;
        const ids = try a.alloc(u32, count);
        var it = view.iterator();
        for (ids) |*id| id.* = it.nextCodepoint().?;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var count: usize = 0;
        var scalar: [4]u8 = undefined;
        for (ids) |id| count += std.unicode.utf8Encode(@intCast(id), &scalar) catch unreachable;
        const decoded = try a.alloc(u8, count);
        var offset: usize = 0;
        for (ids) |id| {
            const size = std.unicode.utf8Encode(@intCast(id), &scalar) catch unreachable;
            @memcpy(decoded[offset..][0..size], scalar[0..size]);
            offset += size;
        }
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(ctx: *anyopaque, a: std.mem.Allocator, messages: json.Value, options: model_text.RenderOptions, problem: *[]const u8) model_text.Error![]u8 {
        const t: *TestText = @ptrCast(@alignCast(ctx));
        if (!t.v41) return a.dupe(u8, "prompt");
        const v41 = @import("deepseek_v41_text.zig");
        const hf = @import("hf_text.zig");
        var context: std.json.ObjectMap = .empty;
        try context.put(a, "enable_thinking", .{ .bool = options.enable_thinking });
        const prepared = try v41.prepare(a, messages, problem);
        return @import("template").renderChat(a, @embedFile("deepseek_v41_official.jinja"), try hf.toStd(a, prepared), if (options.tools) |tools| try hf.toStd(a, tools) else .null, .{ .object = context }, options.add_generation_prompt, null) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            return error.Template;
        };
    }

    fn templateSource(ctx: *anyopaque) []const u8 {
        const t: *TestText = @ptrCast(@alignCast(ctx));
        return if (t.v41) @embedFile("deepseek_v41_official.jinja") else "";
    }
};

/// An 8-token window whose engine answers "hi" at once.
const HiEngine = struct {
    pub fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8 };
    }

    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// Two tool calls byte by byte, with a newline between them that the whole reply does not show.
const CallEngine = struct {
    const reply = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"a\"}}\n</tool_call>\n" ++
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"b\"}}\n</tool_call>";
    const ids = blk: {
        var out: [reply.len]u32 = undefined;
        for (reply, &out) |byte, *id| id.* = byte;
        break :blk out;
    };
    tokens: []const u32 = &ids,
    batched: bool = false,
    prompt: [4096]u32 = undefined,
    prompt_len: usize = 0,

    pub fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = HiEngine.cancel, .status = HiEngine.status, .memory = HiEngine.memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *CallEngine = @ptrCast(@alignCast(ctx));
        e.prompt_len = request.prompt.len;
        @memcpy(e.prompt[0..e.prompt_len], request.prompt);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        if (e.batched) {
            sink.event(sink.ctx, id, &.{ .tokens = e.tokens });
        } else for (0..e.tokens.len) |i| sink.event(sink.ctx, id, &.{ .tokens = e.tokens[i..][0..1] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
};

/// What the route's output saw: a whole reply's status, error code and sampling label, or a stream and its first role.
const Seen = struct {
    gone_at_open: bool = false, // the client is gone when the stream opens
    opened: bool = false,
    events: usize = 0,
    role_first: bool = false,
    status: ?u16 = null,
    context_code: bool = false,
    label: ?enum { exact, greedy } = null,
    preparing: i64 = 0,
    text: [4096]u8 = undefined, // bounded complete content for legacy-envelope regressions
    text_len: usize = 0,
    calls: usize = 0, // the tool calls the stream started, or the whole reply's tool calls
    got_error: bool = false,
    arguments: [1024]u8 = undefined,
    arguments_len: usize = 0,
    argument_fragments: usize = 0,
    reasoning: [256]u8 = undefined,
    reasoning_len: usize = 0,

    fn keepArguments(c: *Seen, call: json.Value) void {
        const f = call.get("function") orelse return;
        const args = f.strField("arguments") orelse return;
        if (args.len > 0) c.argument_fragments += 1;
        const n = @min(args.len, c.arguments.len - c.arguments_len);
        @memcpy(c.arguments[c.arguments_len..][0..n], args[0..n]);
        c.arguments_len += n;
    }

    fn keepReasoning(c: *Seen, value: json.Value) void {
        if (value != .string) return;
        const n = @min(value.string.len, c.reasoning.len - c.reasoning_len);
        @memcpy(c.reasoning[c.reasoning_len..][0..n], value.string[0..n]);
        c.reasoning_len += n;
    }

    pub fn content(c: *const Seen) []const u8 {
        return c.text[0..c.text_len];
    }

    fn keep(c: *Seen, part: json.Value) void {
        if (part != .string) return;
        std.debug.assert(part.string.len <= c.text.len - c.text_len);
        const n = part.string.len;
        @memcpy(c.text[c.text_len..][0..n], part.string[0..n]);
        c.text_len += n;
    }

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        if (c.gone_at_open) return error.Closed;
        c.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.events += 1;
        const p = payload orelse return;
        if (p.get("error")) |err| {
            c.got_error = true;
            const code = err.get("code") orelse return;
            c.context_code = code == .string and std.mem.eql(u8, code.string, "context_length_exceeded");
            return;
        }
        if (p.get("choices")) |ch| if (ch == .array and ch.array.len > 0) if (ch.array[0].get("delta")) |d| {
            if (d.get("content")) |t| c.keep(t);
            if (d.get("reasoning_content")) |t| c.keepReasoning(t);
            if (d.get("tool_calls")) |t| if (t == .array) for (t.array) |call| {
                if (call.get("id") != null) c.calls += 1;
                c.keepArguments(call);
            };
        };
        if (c.events != 1) return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const role = (choices.array[0].get("delta") orelse return).get("role") orelse return;
        c.role_first = role == .string and std.mem.eql(u8, role.string, "assistant");
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        if (payload.get("tensorfold")) |t| if (t.get("sampling")) |l| if (l == .string) {
            c.label = if (std.mem.eql(u8, l.string, "exact")) .exact else if (std.mem.eql(u8, l.string, "greedy")) .greedy else null;
        };
        if (payload.get("choices")) |ch| if (ch == .array and ch.array.len > 0) if (ch.array[0].get("message")) |m| {
            if (m.get("content")) |t| c.keep(t);
            if (m.get("reasoning_content")) |t| c.keepReasoning(t);
            if (m.get("tool_calls")) |t| if (t == .array) {
                c.calls += t.array.len;
                for (t.array) |call| c.keepArguments(call);
            };
        };
        c.got_error = payload.get("error") != null;
        const code = (payload.get("error") orelse return).get("code") orelse return;
        c.context_code = code == .string and std.mem.eql(u8, code.string, "context_length_exceeded");
    }
};

const Route = enum { chat, messages, responses };

/// A Responses request with no stored conversation to read.
const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

/// `body` through `route`'s translation and the chat handler, as the route runs it; the preparing count after it.
fn send(route: Route, body: []const u8) !Seen {
    return sendTo(route, body, .{});
}

fn sendTo(route: Route, body: []const u8, start: Seen) !Seen {
    var backend: HiEngine = .{};
    return sendWith(backend.engine(), route, body, start);
}

fn sendWith(engine: api.Engine, route: Route, body: []const u8, start: Seen) !Seen {
    var text: TestText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, engine, text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    const chat = switch (route) {
        .chat => raw,
        .messages => try anthropic_translate.translate(&cx, raw, false),
        .responses => (try responses_translate.translate(&cx, raw, NoHistory{})).chat,
    };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen = start;
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, chat);
    seen.preparing = srv.preparing.load(.acquire);
    return seen;
}

fn expectContextInStream(seen: Seen, role: bool) !void {
    try std.testing.expect(seen.opened);
    try std.testing.expectEqual(role, seen.role_first);
    try std.testing.expect(seen.context_code);
    try std.testing.expectEqual(@as(?u16, null), seen.status);
    try std.testing.expect(seen.events >= 2);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a streamed chat over the window reports context_length_exceeded after the role chunk" {
    try expectContextInStream(try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}"), true);
}

test "a streamed chat with tools over the window reports the context error with no role chunk" {
    try expectContextInStream(try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]}"), false);
}

test "a streamed Messages request over the window reports the context error in the stream" {
    try expectContextInStream(try send(.messages, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}"), true);
}

test "a streamed Responses request over the window reports the context error in the stream" {
    try expectContextInStream(try send(.responses, "{\"model\":\"test-model\",\"input\":\"x\",\"stream\":true,\"max_output_tokens\":100}"), true);
}

test "a valid streamed chat still opens with the role chunk and gives the preparing count back" {
    const seen = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}");
    try std.testing.expect(seen.opened);
    try std.testing.expect(seen.role_first);
    try std.testing.expect(seen.events >= 3); // the role, the text, the last chunk
    try std.testing.expectEqual(@as(?u16, null), seen.status);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a stream whose client is gone when it opens gives the preparing count back" {
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}";
    const seen = try sendTo(.chat, body, .{ .gone_at_open = true });
    try std.testing.expect(!seen.opened);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a reply labels a greedy request greedy and a sampled one exact" {
    const greedy = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0}");
    try std.testing.expectEqual(@as(?u16, 200), greedy.status);
    try std.testing.expectEqual(.greedy, greedy.label.?);
    const sampled = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0.7,\"seed\":1}");
    try std.testing.expectEqual(.exact, sampled.label.?);
    try std.testing.expectEqual(@as(i64, 0), sampled.preparing);
}

/// A two-call reply on `route`, whole and streamed: the stream sends no text that the whole reply does not show.
fn expectToolStreamLikeWhole(route: Route, comptime fields: []const u8) !void {
    var backend: CallEngine = .{};
    const whole = try sendWith(backend.engine(), route, "{" ++ fields ++ ",\"stream\":false}", .{});
    const streamed = try sendWith(backend.engine(), route, "{" ++ fields ++ ",\"stream\":true}", .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    try std.testing.expectEqual(@as(usize, 2), whole.calls);
    try std.testing.expectEqual(@as(usize, 2), streamed.calls);
    try std.testing.expectEqualStrings(whole.content(), streamed.content());
}

const tool_head = "\"model\":\"test-model\",\"max_tokens\":200,";

test "a streamed chat tool reply sends no whitespace that the whole reply does not show" {
    try expectToolStreamLikeWhole(.chat, tool_head ++ "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]");
}

test "a streamed Messages tool reply sends no whitespace that the whole reply does not show" {
    try expectToolStreamLikeWhole(.messages, tool_head ++ "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]");
}

test "a streamed Responses tool reply sends no whitespace that the whole reply does not show" {
    try expectToolStreamLikeWhole(.responses, tool_head ++ "\"input\":\"x\"," ++
        "\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]");
}

test "malformed and unoffered V41 DSML fails both whole and streamed chat" {
    const inputs = [_][]const u8{
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">bad</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"not_offered\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    };
    for (inputs) |raw| {
        const tokens = try TestText.encode(undefined, std.testing.allocator, raw, false);
        defer std.testing.allocator.free(tokens);
        var backend: CallEngine = .{ .tokens = tokens };
        const body = "{\"model\":\"test-model\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
            "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
        const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
        const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
        try std.testing.expectEqual(@as(?u16, 500), whole.status);
        try std.testing.expect(whole.got_error and streamed.got_error);
        try std.testing.expectEqual(@as(usize, 0), whole.calls);
        try std.testing.expectEqual(@as(usize, 0), streamed.calls);
    }
}

test "legacy calls inside V41 DSML parameters stay literal in whole and streamed replies" {
    const literal = "<tool_call><function=lookup></function></tool_call>";
    const raw = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n" ++
        "<｜DSML｜ parameter name=\"q\" string=\"true\">" ++ literal ++ "</｜DSML｜ parameter>\n" ++
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const tokens = try TestText.encode(undefined, std.testing.allocator, raw, false);
    defer std.testing.allocator.free(tokens);
    const body = "{\"model\":\"test-model\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    for ([_]bool{ false, true }) |batched| {
        var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
        const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
        const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(!whole.got_error and !streamed.got_error);
        try std.testing.expectEqual(@as(usize, 1), whole.calls);
        try std.testing.expectEqual(@as(usize, 1), streamed.calls);
        try std.testing.expectEqualStrings("{\"q\":\"" ++ literal ++ "\"}", whole.arguments[0..whole.arguments_len]);
        try std.testing.expectEqualStrings(whole.arguments[0..whole.arguments_len], streamed.arguments[0..streamed.arguments_len]);
    }
}

test "complete V41 calls recovered from unclosed thinking agree in both reply modes" {
    const raw = "<think>reason\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const tokens = try TestText.encode(undefined, std.testing.allocator, raw, false);
    defer std.testing.allocator.free(tokens);
    var backend: CallEngine = .{ .tokens = tokens };
    const body = "{\"model\":\"test-model\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"chat_template_kwargs\":{\"enable_thinking\":true}," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
    const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    try std.testing.expect(!whole.got_error and !streamed.got_error);
    try std.testing.expectEqual(@as(usize, 1), whole.calls);
    try std.testing.expectEqual(@as(usize, 1), streamed.calls);
    try std.testing.expectEqualStrings("{}", whole.arguments[0..whole.arguments_len]);
    try std.testing.expectEqualStrings("{}", streamed.arguments[0..streamed.arguments_len]);
    try std.testing.expectEqualStrings("reason", whole.reasoning[0..whole.reasoning_len]);
    try std.testing.expectEqualStrings("reason", @import("reply_text.zig").pyStrip(streamed.reasoning[0..streamed.reasoning_len]));
    try std.testing.expectEqualStrings(whole.content(), streamed.content());
}

test "Anthropic V41 counting and generation preserve the same multipart prompt" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: TestText = .{ .v41 = true };
    var backend: CallEngine = .{ .tokens = &.{ 'h', 'i' } };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const body = (try json.parse(a, "{\"model\":\"test-model\",\"max_tokens\":2,\"thinking\":{\"type\":\"disabled\"}," ++
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"one\"},{\"type\":\"text\",\"text\":\"two\"}]}]}")).ok;
    const count_chat = try anthropic_translate.translate(&cx, body, true);
    const count = try @import("anthropic.zig").countTokens(srv, &cx, count_chat);
    var seen: Seen = .{};
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, try anthropic_translate.translate(&cx, body, false));
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    try std.testing.expectEqual(count, backend.prompt_len);
    const expected = "<｜begin▁of▁sentence｜><｜User｜>one\n\ntwo<｜Assistant｜></think>";
    try std.testing.expectEqualStrings(expected, try TestText.decode(undefined, a, backend.prompt[0..backend.prompt_len]));
}

test "valid V41 DSML has the same call and prose in whole and streamed chat" {
    const raw = "summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n" ++
        "<｜DSML｜ parameter name=\"image\" string=\"true\">cat.jpg</｜DSML｜ parameter>\n" ++
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const tokens = try TestText.encode(undefined, std.testing.allocator, raw, false);
    defer std.testing.allocator.free(tokens);
    var backend: CallEngine = .{ .tokens = tokens };
    const body = "{\"model\":\"test-model\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
    const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    try std.testing.expect(!whole.got_error and !streamed.got_error);
    try std.testing.expectEqual(@as(usize, 1), whole.calls);
    try std.testing.expectEqual(@as(usize, 1), streamed.calls);
    try std.testing.expectEqualStrings("summary", whole.content());
    try std.testing.expectEqualStrings(whole.content(), streamed.content());
}

/// HiEngine's replies from an engine that decodes greedily only.
const GreedyEngine = struct {
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = HiEngine.submit, .cancel = HiEngine.cancel, .status = HiEngine.status, .memory = HiEngine.memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .greedy_only = true, .thinking_budget = false };
    }
};

test "unsupported thinking budgets are refused before opening a chat stream" {
    var backend: GreedyEngine = .{};
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"max_tokens\":2,\"temperature\":0,\"thinking_budget\":5,\"chat_template_kwargs\":{\"enable_thinking\":true}";
    const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
    const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
    try std.testing.expectEqual(@as(?u16, 400), whole.status);
    try std.testing.expectEqual(@as(?u16, 400), streamed.status);
    try std.testing.expect(!streamed.opened);
    try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
    try @import("admission_consumer_regressions.zig").budgetWithoutThinking(backend.engine(), sendWith);
}

test "a greedy-only engine's sampled request is a 400 before the stream opens; greedy ones are served" {
    var backend: GreedyEngine = .{};
    const sampled = try sendWith(backend.engine(), .chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2,\"temperature\":0.7}", .{});
    try std.testing.expect(!sampled.opened);
    try std.testing.expectEqual(@as(?u16, 400), sampled.status);
    try std.testing.expectEqual(@as(i64, 0), sampled.preparing);
    const greedy = try sendWith(backend.engine(), .chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0}", .{});
    try std.testing.expectEqual(@as(?u16, 200), greedy.status);
    try std.testing.expectEqualStrings("hi", greedy.content());
    const unset = try sendWith(backend.engine(), .chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2}", .{});
    try std.testing.expectEqual(@as(?u16, 200), unset.status);
}

test "a greedy-only engine's refusal names --temperature 0 when the server's own default samples" {
    var backend: GreedyEngine = .{};
    var text: TestText = .{};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { []const u8, bool }{ .{ "{\"temperature\":1.0}", true }, .{ "{\"temperature\":0}", false } }) |case| {
        const config: server.Config = .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .default_sampling = (try json.parse(a, case[0])).ok };
        var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), config, null);
        defer srv.deinit();
        var cx: errors.Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, srv.resolveSampling(&cx, (try json.parse(a, "{\"temperature\":0.7}")).ok, 0, &.{}));
        try std.testing.expectEqual(case[1], std.mem.indexOf(u8, cx.message, "--temperature 0") != null);
        try std.testing.expect(try srv.resolveSampling(&cx, (try json.parse(a, "{\"temperature\":0}")).ok, 0, &.{}) == null);
    }
}

test "nested legacy markup in malformed V41 DSML is never executable" {
    const prefix = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n" ++
        "<｜DSML｜ parameter name=\"q\" string=\"true\"><tool_call><function=lookup></function></tool_call>";
    const inputs = [_][]const u8{
        prefix,
        prefix ++ "</｜DSML｜ parameter>\n</｜DSML｜ calls>",
        "<think>reason\n" ++ prefix ++ "</｜DSML｜ parameter>\n</｜DSML｜ calls>",
        "<think>reason\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"not_offered\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    };
    for (inputs, 0..) |raw, i| {
        const tokens = try TestText.encode(undefined, std.testing.allocator, raw, false);
        defer std.testing.allocator.free(tokens);
        var backend: CallEngine = .{ .tokens = tokens };
        const body = try std.fmt.allocPrint(std.testing.allocator, "{{\"model\":\"test-model\",\"max_tokens\":1000,\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]," ++
            "\"chat_template_kwargs\":{{\"enable_thinking\":{s}}}," ++
            "\"tools\":[{{\"type\":\"function\",\"function\":{{\"name\":\"lookup\",\"parameters\":{{\"type\":\"object\"}}}}}}]", .{if (i >= 2) "true" else "false"});
        defer std.testing.allocator.free(body);
        const whole_body = try std.mem.concat(std.testing.allocator, u8, &.{ body, ",\"stream\":false}" });
        defer std.testing.allocator.free(whole_body);
        const streamed_body = try std.mem.concat(std.testing.allocator, u8, &.{ body, ",\"stream\":true}" });
        defer std.testing.allocator.free(streamed_body);
        const whole = try sendWith(backend.engine(), .chat, whole_body, .{});
        const streamed = try sendWith(backend.engine(), .chat, streamed_body, .{});
        try std.testing.expectEqual(@as(?u16, 500), whole.status);
        try std.testing.expect(whole.got_error and streamed.got_error);
        try std.testing.expectEqual(@as(usize, 0), whole.calls);
        try std.testing.expectEqual(@as(usize, 0), streamed.calls);
    }
}

test "legacy XML recovered from unclosed thinking retains its whole-arguments fallback" {
    const raw = "reason\n<tool_call>\n<function=lookup>\n<parameter=query>\nfast cars\n</parameter>\n" ++
        "<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>";
    const tokens = try TestText.encode(undefined, std.testing.allocator, raw, false);
    defer std.testing.allocator.free(tokens);
    var backend: CallEngine = .{ .tokens = tokens };
    const body = "{\"model\":\"test-model\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"chat_template_kwargs\":{\"enable_thinking\":true}," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
        "\"properties\":{\"limit\":{\"type\":\"integer\"}}}}}]";
    const whole = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":false}", .{});
    const streamed = try sendWith(backend.engine(), .chat, body ++ ",\"stream\":true}", .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    try std.testing.expect(!whole.got_error and !streamed.got_error);
    try std.testing.expectEqual(@as(usize, 1), whole.calls);
    try std.testing.expectEqual(@as(usize, 1), streamed.calls);
    try std.testing.expectEqualStrings("{\"query\":\"fast cars\",\"limit\":5}", whole.arguments[0..whole.arguments_len]);
    try std.testing.expectEqualStrings(whole.arguments[0..whole.arguments_len], streamed.arguments[0..streamed.arguments_len]);
    try std.testing.expectEqual(@as(usize, 1), streamed.argument_fragments);
}
test "containment final-review legacy consumer regressions remain supported" {
    try @import("legacy_consumer_regressions.zig").run(TestText, CallEngine, sendWith);
    try @import("legacy_consumer_regressions.zig").mixedChain(TestText, CallEngine, sendWith);
    try @import("decoder_consumer_regressions.zig").run(TestText, HiEngine);
}
test "vision input and audio output are refused before opening a stream" {
    try @import("admission_consumer_regressions.zig").media(send);
}
test "explicit drafts are refused rather than ignored when server drafts are disabled" {
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"max_tokens\":2,\"draft\":true";
    for ([_][]const u8{ ",\"stream\":false}", ",\"stream\":true}" }) |suffix| {
        const raw = try std.mem.concat(std.testing.allocator, u8, &.{ body, suffix });
        defer std.testing.allocator.free(raw);
        const seen = try send(.chat, raw);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(!seen.opened);
    }
}
