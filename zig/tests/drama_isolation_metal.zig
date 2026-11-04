//! Cooperative distinct-history state isolation; not HTTP, simultaneous execution or independent fidelity.
const std = @import("std");
const d = @import("drama3");
const E = d.Engine;
const W = d.config.frame_width;
pub const History = struct { rows: []const [W]i32, forced: []const [W]i32 };

// The load planner has a fixed scratch margin, not a row-scaled high-water estimate.
pub fn validateChunk(chunk: usize) !void {
    if (chunk == 0 or chunk > 128) return error.BadIsolationChunk;
}

test "live byte comparator isolation caps row scratch at tested chunk width" {
    try validateChunk(1);
    try validateChunk(128);
    try std.testing.expectError(error.BadIsolationChunk, validateChunk(0));
    try std.testing.expectError(error.BadIsolationChunk, validateChunk(129));
    try std.testing.expectError(error.BadIsolationChunk, validateChunk(8192));
}

pub fn readRows(a: std.mem.Allocator, io: std.Io, path: []const u8) ![][W]i32 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(8192 * W * 4));
    defer a.free(raw);
    if (raw.len == 0 or raw.len % (W * 4) != 0) return error.BadIsolationRows;
    const rows = try a.alloc([W]i32, raw.len / (W * 4));
    @memcpy(std.mem.sliceAsBytes(rows), raw);
    return rows;
}

pub fn validate(a: std.mem.Allocator, h: History, ids: d.config.Ids, context: usize) !void {
    if (context < 1025 or context > 8192 or h.rows.len < 2 or h.forced.len == 0) return error.BadIsolationWindow;
    if (h.rows.len > context - 2 or h.forced.len > context - 2 - h.rows.len) return error.BadIsolationWindow;
    const joined = try a.alloc([W]i32, h.rows.len + h.forced.len);
    defer a.free(joined);
    @memcpy(joined[0..h.rows.len], h.rows);
    @memcpy(joined[h.rows.len..], h.forced);
    try d.model.validate(joined, ids, context, 1);
    for (h.forced) |row| if (row[0] < ids.sem_begin or row[0] > ids.sem_end) return error.ForcedAcousticRowRequired;
}

pub fn distinct(a: History, b: History) !void {
    if (std.mem.eql(u8, std.mem.sliceAsBytes(a.rows), std.mem.sliceAsBytes(b.rows)) or
        std.mem.eql(u8, std.mem.sliceAsBytes(a.forced), std.mem.sliceAsBytes(b.forced)))
        return error.IsolationHistoriesMustDiffer;
}

test "live byte comparator isolation requires distinct prompts and forced histories" {
    const a = [_][W]i32{@splat(0)};
    const b = [_][W]i32{@splat(1)};
    try distinct(.{ .rows = &a, .forced = &a }, .{ .rows = &b, .forced = &b });
    try std.testing.expectError(error.IsolationHistoriesMustDiffer, distinct(.{ .rows = &a, .forced = &a }, .{ .rows = &a, .forced = &b }));
    try std.testing.expectError(error.IsolationHistoriesMustDiffer, distinct(.{ .rows = &a, .forced = &a }, .{ .rows = &b, .forced = &a }));
}

const Lane = struct {
    e: *E,
    h: History,
    frame: u64,
    picks: u64,
    at: usize = 0,
    made: usize = 0,

    fn init(e: *E, h: History) !Lane {
        return .{ .e = e, .h = h, .frame = try e.scratch(W * 4), .picks = try e.scratch(W * 4) };
    }

    fn begin(l: *Lane, poison: bool) !void {
        l.at = 0;
        l.made = 0;
        if (poison) {
            const n = (l.h.rows.len + l.h.forced.len) * d.config.kv_heads * d.config.head_dim * 2;
            const bytes = try l.e.gpa.alloc(u8, n);
            defer l.e.gpa.free(bytes);
            @memset(bytes, 0xa5);
            for (0..d.config.layers) |i| if (l.e.c.full_attention[i]) {
                for (0..2) |kv| try l.e.put(l.e.kv_bb[i][kv], bytes);
            };
            const tower = n - d.config.kv_heads * d.config.head_dim * 2;
            for (0..d.config.tower_layers) |i| for (0..2) |kv| {
                try l.e.put(l.e.kv_tw[i][kv], bytes[0..tower]);
            };
        }
        try l.e.startPrompt(l.h.rows);
    }

    fn done(l: *const Lane) bool {
        return l.at == l.h.rows.len and l.made == l.h.forced.len;
    }

    fn advance(l: *Lane, chunk: usize) !void {
        if (l.at < l.h.rows.len) {
            const count = @min(chunk, l.h.rows.len - l.at);
            try l.e.promptChunk(l.h.rows.len, l.at, count, null);
            l.at += count;
            if (l.at == l.h.rows.len) try l.e.finishPrompt(l.h.rows);
        } else if (l.made < l.h.forced.len) {
            try l.e.put(l.frame, std.mem.asBytes(&l.h.forced[l.made]));
            try l.e.frameForced(.{ .frame = l.frame, .picks = l.picks }, null);
            l.made += 1;
        }
        try l.e.flush();
    }
};

fn compareLane(comptime Compare: type, reference: Lane, observed: Lane, name: []const u8, order: []const u8) !void {
    const total = reference.h.rows.len + reference.h.forced.len;
    var c: Compare = .{ .a = reference.e, .b = observed.e, .seam = total };
    compareState(Compare, &c, reference.h) catch |err| {
        std.debug.print("MISMATCH cooperative-isolation lane={s} order={s} error={s}\n", .{ name, order, @errorName(err) });
        return err;
    };
    std.debug.print("PASS cooperative-isolation lane={s} order={s} prompt_rows={d} forced_rows={d} buffers={d} bytes={d} poison_kv=true; terminal state only, no HTTP, simultaneous, CFG, sampler or independent fidelity qualification\n", .{ name, order, reference.h.rows.len, reference.h.forced.len, c.buffers, c.bytes });
}

fn compareState(comptime Compare: type, c: *Compare, h: History) !void {
    try c.live(h.rows.len + h.forced.len);
    try c.final();
    try c.range("forced_outputs", 0, c.a.frames_out, c.b.frames_out, h.forced.len * W * 4);
    const emitted = try c.a.gpa.alloc([W]i32, h.forced.len);
    defer c.a.gpa.free(emitted);
    try c.a.download(c.a.frames_out, std.mem.sliceAsBytes(emitted));
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(emitted), std.mem.sliceAsBytes(h.forced))) return error.IsolationForcedOutputMismatch;
}

pub fn run(comptime Compare: type, base: *E, candidate: *E, a: History, b: History, chunk: usize) !void {
    const lane_a = try base.fork();
    defer lane_a.destroy();
    const lane_b = try base.fork();
    defer lane_b.destroy();
    var solo_a = try Lane.init(base, a);
    var solo_b = try Lane.init(candidate, b);
    var inter_a = try Lane.init(lane_a, a);
    var inter_b = try Lane.init(lane_b, b);
    try solo_a.begin(true);
    while (!solo_a.done()) try solo_a.advance(chunk);
    try solo_b.begin(true);
    while (!solo_b.done()) try solo_b.advance(chunk);
    for ([_]bool{ false, true }) |reverse| {
        try inter_a.begin(true);
        try inter_b.begin(true);
        while (!inter_a.done() or !inter_b.done()) {
            if (reverse) {
                if (!inter_b.done()) try inter_b.advance(chunk);
                if (!inter_a.done()) try inter_a.advance(chunk);
            } else {
                if (!inter_a.done()) try inter_a.advance(chunk);
                if (!inter_b.done()) try inter_b.advance(chunk);
            }
        }
        const order = if (reverse) "BA" else "AB";
        try compareLane(Compare, solo_a, inter_a, "A", order);
        try compareLane(Compare, solo_b, inter_b, "B", order);
    }
}
test "live byte comparator isolation validates windows before allocating" {
    const ids: d.config.Ids = .{ .eos = 1, .pad = 2, .sem_begin = 100, .sem_end = 4195, .audio_pad = 3 };
    try std.testing.expectError(error.BadIsolationWindow, validate(std.testing.allocator, .{ .rows = &.{}, .forced = &.{} }, ids, 8192));
}
