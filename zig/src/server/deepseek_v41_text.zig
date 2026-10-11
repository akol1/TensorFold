//! Text-only official DS4.1 template adaptation; pinned DeepSeek-AI revision and MIT notice in THIRD_PARTY_NOTICES.md.
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Error = error{ Template, OutOfMemory };

pub fn isTemplate(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "DeepSeek-V4.1 chat template") != null and
        std.mem.indexOf(u8, source, " calls>") != null;
}

fn rejectMedia(value: Value) Error!void {
    switch (value) {
        .string => |s| if (std.mem.indexOf(u8, s, "<｜deepseek_image｜>") != null or
            std.mem.indexOf(u8, s, "<image>") != null or std.mem.indexOf(u8, s, "</image>") != null) return error.Template,
        .array => |items| for (items) |item| try rejectMedia(item),
        .object => |o| {
            for ([_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" }) |name| {
                if (value.typeIs(name) or json.truthyField(value, name)) return error.Template;
            }
            for (o.values()) |item| try rejectMedia(item);
        },
        else => {},
    }
}

fn validText(value: ?Value) Error!void {
    const v = value orelse return;
    if (v == .null or v == .string) return;
    if (v != .array) return error.Template;
    for (v.array) |part| {
        if (part != .object or part.strField("type") == null) return error.Template;
        if (part.typeIs("text")) {
            if (part.field("text")) |t| if (t != .string) return error.Template;
        } else if (part.typeIs("tool_result")) try validText(part.get("content"));
    }
}

fn validateMessage(message: Value) Error!void {
    if (message != .object) return error.Template;
    for ([_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" }) |name| {
        if (json.truthyField(message, name)) return error.Template;
    }
    for ([_][]const u8{ "content", "content_blocks", "reasoning_content", "thinking" }) |name| {
        if (message.get(name)) |value| try rejectMedia(value);
    }
    try validText(message.get("content"));
    try validText(message.get("content_blocks"));
    const role = message.strField("role") orelse return error.Template;
    for ([_][]const u8{ "system", "developer", "user", "assistant", "tool", "latest_reminder" }) |accepted| {
        if (std.mem.eql(u8, role, accepted)) return;
    }
    return error.Template;
}

pub fn validate(messages: Value, problem: *[]const u8) Error!void {
    if (messages != .array or messages.array.len == 0) {
        problem.* = "DeepSeek V4.1 messages must be a nonempty array";
        return error.Template;
    }
    for (messages.array) |message| validateMessage(message) catch |err| {
        problem.* = "DeepSeek V4.1 requires text-only messages with a supported role";
        return err;
    };
}

/// Preserve multipart separators, leading/mid systems and metadata; only normalize API aliases and call arguments.
pub fn prepare(a: Allocator, messages: Value, problem: *[]const u8) Error!Value {
    return prepareImpl(a, messages) catch |err| {
        if (err == error.Template)
            problem.* = "DeepSeek V4.1 requires text-only messages with supported roles; images, audio and video are unsupported";
        return err;
    };
}

fn prepareImpl(a: Allocator, messages: Value) Error!Value {
    if (messages != .array or messages.array.len == 0) return error.Template;
    const list = try a.alloc(Value, messages.array.len);
    for (messages.array, list) |message, *dst| {
        try validateMessage(message);
        const role = message.strField("role").?;
        const copy = try json.copyObject(a, message.object);
        if (std.mem.eql(u8, role, "developer")) try copy.put(a, "role", .{ .string = "system" });
        if (message.field("content_blocks")) |blocks| {
            try validText(blocks);
            try copy.put(a, "content", blocks);
        }
        if (message.field("tool_calls")) |calls| {
            if (calls != .array) return error.Template;
            const changed = try a.alloc(Value, calls.array.len);
            for (calls.array, changed) |call, *slot| {
                const function = call.get("function") orelse return error.Template;
                if (function != .object or function.strField("name") == null) return error.Template;
                const original = function.get("arguments") orelse return error.Template;
                var arguments = original;
                for (0..2) |_| {
                    if (arguments != .string) break;
                    arguments = switch (try json.parseText(a, arguments.string)) {
                        .ok => |v| v,
                        .err => break,
                    };
                }
                if (arguments != .object) {
                    const wrapped = try json.newObject(a);
                    try wrapped.put(a, "arguments", original);
                    arguments = .{ .object = wrapped };
                }
                const fn_copy = try json.copyObject(a, function.object);
                try fn_copy.put(a, "arguments", arguments);
                const call_copy = try json.copyObject(a, call.object);
                try call_copy.put(a, "function", .{ .object = fn_copy });
                slot.* = .{ .object = call_copy };
            }
            try copy.put(a, "tool_calls", .{ .array = changed });
        }
        dst.* = .{ .object = copy };
    }
    return .{ .array = list };
}

/// Keep official aliases; server xhigh means max, and templates without medium use high.
pub fn effort(a: Allocator, source: []const u8, input: []const u8) Error!std.json.Value {
    if (std.fmt.parseInt(i64, input, 10)) |n| {
        if (n < 1 or n > 100) return error.Template;
        return .{ .integer = n };
    } else |_| {}
    if (std.mem.eql(u8, input, "xhigh")) return .{ .string = "max" };
    if (std.mem.eql(u8, input, "minimal")) return .{ .string = "low" };
    if (std.mem.eql(u8, input, "medium") and std.mem.indexOf(u8, source, "\"medium\"") == null) return .{ .string = "high" };
    if (std.mem.eql(u8, input, "none")) return .{ .string = "high" };
    _ = a;
    return .{ .string = input };
}
