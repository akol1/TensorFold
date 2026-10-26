const std = @import("std");
const json = @import("json");
const Value = json.Value;

pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };

/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
    /// ``logprobs: true`` with this many ``top_logprobs``; null: not asked.
    logprobs: ?u8 = null,
    /// ``/v1/completions``' ``prompt_logprobs``: rows for the prompt with this many alternatives; null: not asked.
    prompt_logprobs: ?u8 = null,
};
