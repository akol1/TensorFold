const std = @import("std");
const builtin = @import("builtin");
const input_mod = @import("input.zig");
const Allocator = std.mem.Allocator;
const Input = input_mod.Input;

/// The request number in a dump's name: the benchmark's fixed request order lines two runs' dumps up.
var seq: std.atomic.Value(u32) = .init(0);

/// TF_DUMP_IDS=<dir>: write prompt and output ids as JSON arrays named <request>-<slug>-prompt/-out.json; never fatal.
pub fn ids(io: std.Io, a: Allocator, input: Input, prompt_ids: []const u32, out_ids: []const u32) void {
    const raw = std.c.getenv("TF_DUMP_IDS") orelse return;
    const dir = std.mem.span(raw);
    if (dir.len == 0) return;
    // a dump directory is private on POSIX; Windows keeps the default attributes, as it has no mode bits
    const permissions: std.Io.Dir.Permissions = if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700);
    _ = std.Io.Dir.cwd().createDirPathStatus(io, dir, permissions) catch {};
    const n = seq.fetchAdd(1, .monotonic) + 1;
    const slug = promptSlug(a, input) catch "";
    const stem = if (slug.len > 0)
        std.fmt.allocPrint(a, "{d:0>4}-{s}", .{ n, slug }) catch return
    else
        std.fmt.allocPrint(a, "{d:0>4}", .{n}) catch return;
    writeIds(io, a, dir, stem, "prompt", prompt_ids);
    writeIds(io, a, dir, stem, "out", out_ids);
}

/// A request's own prompt words as a filename: no request carries a fixture name, so the text tells fixtures apart.
fn promptSlug(a: Allocator, input: Input) error{ OutOfMemory, WriteFailed }![]const u8 {
    const text: []const u8 = blk: {
        if (input.prompt) |p| switch (p) {
            .text => |t| break :blk t,
            .ids => break :blk "",
        };
        var found: []const u8 = "";
        switch (input.messages) {
            .array => |messages| for (messages) |m| {
                const role = m.get("role") orelse continue;
                if (role == .string and !std.mem.eql(u8, role.string, "user")) continue;
                const content = m.get("content") orelse continue;
                if (content == .string) found = content.string;
            },
            else => {},
        }
        break :blk found;
    };
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var last_dash = false;
    for (text) |ch| {
        if (out.written().len >= 28) break;
        if (std.ascii.isAlphanumeric(ch)) {
            try out.writer.writeByte(std.ascii.toLower(ch));
            last_dash = false;
        } else if (!last_dash and out.written().len > 0) {
            try out.writer.writeByte('-');
            last_dash = true;
        }
    }
    const slug = out.written();
    const end = std.mem.trimEnd(u8, slug, "-").len;
    return try a.dupe(u8, slug[0..end]);
}

/// One dump file: ``[1,2,3]`` and a newline.
fn writeIds(io: std.Io, a: Allocator, dir: []const u8, stem: []const u8, kind: []const u8, list: []const u32) void {
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    out.writer.writeByte('[') catch return;
    for (list, 0..) |t, i| {
        if (i > 0) out.writer.writeByte(',') catch return;
        out.writer.print("{d}", .{t}) catch return;
    }
    out.writer.writeAll("]\n") catch return;
    const name = std.fmt.allocPrint(a, "{s}-{s}.json", .{ stem, kind }) catch return;
    const path = std.fs.path.join(a, &.{ dir, name }) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.written() }) catch {};
}
