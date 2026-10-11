//! The request's stop strings as the engine checks each newly emitted token.
const std = @import("std");
const Server = @import("server.zig").Server;
const reply_text = @import("reply_text.zig");
const api = @import("engine_api");

pub const Hook = struct {
    srv: *Server,
    stops: reply_text.Stops,

    pub fn check(ctx: *anyopaque, emitted: []const u32) api.Stop.Error!bool {
        const h: *Hook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        const tail = emitted[emitted.len -| h.stops.tail()..];
        const text = try h.srv.text.decode(arena.allocator(), tail);
        for (h.stops.strings) |stop| if (std.mem.indexOf(u8, text, stop) != null) return true;
        return false;
    }
};
