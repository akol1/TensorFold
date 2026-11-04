//! Exact live-state comparison of Metal prompt chunks; no audio or independent fidelity claim.
const std = @import("std");
const d = @import("drama3");
const C = d.config;
const E = d.Engine;
const W = C.frame_width;
const isolation = @import("drama_isolation_metal.zig");
test {
    _ = isolation;
}

fn firstDifference(a: []const u8, b: []const u8) ?usize {
    for (a, b, 0..) |x, y, i| if (x != y) return i;
    return null;
}

test "live byte comparator detects mutation and excludes unwritten tails" {
    const a = [_]u8{ 1, 2, 3, 99 };
    var b = [_]u8{ 1, 2, 3, 0 };
    try std.testing.expectEqual(null, firstDifference(a[0..3], b[0..3]));
    b[1] ^= 1;
    try std.testing.expectEqual(@as(?usize, 1), firstDifference(a[0..3], b[0..3]));
}

pub const Compare = struct {
    a: *E,
    b: *E,
    buffers: usize = 0,
    bytes: usize = 0,
    seam: usize = 0,

    pub fn range(c: *Compare, name: []const u8, layer: usize, pa: u64, pb: u64, n: usize) !void {
        if (n == 0) return;
        const a = try c.a.gpa.alloc(u8, n);
        defer c.a.gpa.free(a);
        const b = try c.a.gpa.alloc(u8, n);
        defer c.a.gpa.free(b);
        try c.a.download(pa, a);
        try c.b.download(pb, b);
        c.buffers += 1;
        c.bytes += n;
        if (firstDifference(a, b)) |i| {
            std.debug.print("MISMATCH seam={d} state={s} layer={d} byte={d} reference={x} candidate={x} buffers={d} bytes={d}\n", .{ c.seam, name, layer, i, a[i], b[i], c.buffers, c.bytes });
            return error.ChunkStateMismatch;
        }
    }

    pub fn live(c: *Compare, total: usize) !void {
        try c.a.flush();
        try c.b.flush();
        const stride = C.kv_heads * C.head_dim * 2;
        for (0..2) |i| try c.range("recurrent", i, c.a.state_bufs[i], c.b.state_bufs[i], c.a.state_bytes[i]);
        for (0..C.layers) |li| {
            if (!c.a.c.full_attention[li]) continue;
            for (0..2) |kv| try c.range(if (kv == 0) "backbone.K" else "backbone.V", li, c.a.kv_bb[li][kv], c.b.kv_bb[li][kv], c.seam * stride);
        }
        for (0..C.tower_layers) |li| {
            for (0..2) |kv| try c.range(if (kv == 0) "tower.K" else "tower.V", li, c.a.kv_tw[li][kv], c.b.kv_tw[li][kv], @min(c.seam, total - 1) * stride);
        }
    }

    pub fn final(c: *Compare) !void {
        try c.a.flush();
        try c.b.flush();
        inline for (.{ .{ "hlast", C.hidden * 2 }, .{ "cur_input", W * 4 }, .{ "window", 16 * 4 }, .{ "pos", 4 }, .{ "step", 8 }, .{ "count", 4 } }) |item|
            try c.range(item[0], 0, @field(c.a, item[0]), @field(c.b, item[0]), item[1]);
        if (c.a.rows_done != c.b.rows_done or c.a.prompt_rows != c.b.prompt_rows) {
            std.debug.print("MISMATCH seam={d} state=logical_rows reference={d}/{d} candidate={d}/{d} buffers={d} bytes={d}\n", .{ c.seam, c.a.rows_done, c.a.prompt_rows, c.b.rows_done, c.b.prompt_rows, c.buffers, c.bytes });
            return error.LogicalRowMismatch;
        }
    }
};

fn validateResumeCut(rows: usize, cut: usize, chunk: usize) !void {
    if (rows < 3 or cut == 0 or cut >= rows - 1 or chunk == 0 or cut % chunk != 0)
        return error.BadResumeCut;
}

test "live byte comparator resume cut bounds preserve tower lookahead" {
    try validateResumeCut(24, 3, 3);
    try validateResumeCut(257, 128, 128);
    try validateResumeCut(24, 22, 1);
    for ([_]usize{ 0, 23, 24, std.math.maxInt(usize) }) |cut|
        try std.testing.expectError(error.BadResumeCut, validateResumeCut(24, cut, 1));
    try std.testing.expectError(error.BadResumeCut, validateResumeCut(24, 4, 3));
    try std.testing.expectError(error.BadResumeCut, validateResumeCut(24, 3, 0));
    try std.testing.expectError(error.BadResumeCut, validateResumeCut(2, 1, 1));
}

const PrefixDigest = struct {
    hash: [32]u8,
    bytes: usize,
};

fn prefixKvDigest(e: *E, cut: usize) !PrefixDigest {
    try e.flush();
    const n = cut * C.kv_heads * C.head_dim * 2;
    const bytes = try e.gpa.alloc(u8, n);
    defer e.gpa.free(bytes);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var total: usize = 0;
    for (0..C.layers) |li| {
        if (!e.c.full_attention[li]) continue;
        for (0..2) |kv| {
            try e.download(e.kv_bb[li][kv], bytes);
            hash.update(bytes);
            total += n;
        }
    }
    for (0..C.tower_layers) |li| {
        for (0..2) |kv| {
            try e.download(e.kv_tw[li][kv], bytes);
            hash.update(bytes);
            total += n;
        }
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return .{ .hash = digest, .bytes = total };
}

fn verifyPrefixKv(e: *E, cut: usize, expected: PrefixDigest) !void {
    const got = try prefixKvDigest(e, cut);
    if (got.bytes != expected.bytes or !std.mem.eql(u8, &got.hash, &expected.hash))
        return error.ResumedPrefixKvChanged;
}

fn promptRange(e: *E, total: usize, start: usize, end: usize, chunk: usize) !void {
    var at = start;
    while (at < end) {
        const count = @min(chunk, end - at);
        try e.promptChunk(total, at, count, null);
        at += count;
    }
}

fn resumedState(base: *E, candidate: *E, rows: []const [W]i32, cut: usize, chunk: usize) !void {
    try validateResumeCut(rows.len, cut, chunk);
    try candidate.startPrompt(rows);
    try promptRange(candidate, rows.len, 0, cut, chunk);
    var saved: E.Snapshot = .{};
    try candidate.snapshot(&saved);
    const prefix = try prefixKvDigest(candidate, cut);
    // Complete the suffix to dirty recurrent state while retaining the live prefix KV in this lane.
    try promptRange(candidate, rows.len, cut, rows.len, chunk);
    try candidate.finishPrompt(rows);
    try verifyPrefixKv(candidate, cut, prefix);
    try candidate.restore(&saved);
    try candidate.beginPromptFrom(rows, cut);
    try verifyPrefixKv(candidate, cut, prefix);
    try promptRange(candidate, rows.len, cut, rows.len, chunk);
    try candidate.finishPrompt(rows);
    try verifyPrefixKv(candidate, cut, prefix);
    var resumed: Compare = .{ .a = base, .b = candidate, .seam = rows.len };
    try resumed.live(rows.len);
    try resumed.final();
    std.debug.print("PASS resumed-state rows={d} cut={d} chunk={d} prefix_kv_bytes={d} prefix_kv_checks=3 buffers={d} bytes={d}; retained KV, no persistence or independent fidelity qualification\n", .{ rows.len, cut, chunk, prefix.bytes, resumed.buffers, resumed.bytes });
}

fn forcedWindow(prefix: usize, forced: usize, context: usize) !usize {
    if (context < 1025 or context > 8192 or prefix < 2 or forced == 0) return error.BadForcedWindow;
    if (prefix > context - 2 or forced > context - 2 - prefix) return error.BadForcedWindow;
    return prefix + forced;
}

test "live byte comparator forced history window bounds" {
    try std.testing.expectEqual(@as(usize, 8190), try forcedWindow(24, 8166, 8192));
    try std.testing.expectError(error.BadForcedWindow, forcedWindow(24, 8167, 8192));
    try std.testing.expectError(error.BadForcedWindow, forcedWindow(24, 0, 8192));
    try std.testing.expectError(error.BadForcedWindow, forcedWindow(8193, 1, 8192));
}

const LogitProbe = struct {
    const total = 1 + C.tower_layers + C.codebooks - 1;
    engine: *E,
    reference: ?*const LogitProbe = null,
    saved: [total]?[]u8 = @splat(null),
    next: usize = 0,
    bytes: usize = 0,

    fn deinit(p: *LogitProbe) void {
        for (p.saved) |bytes| if (bytes) |b| p.engine.gpa.free(b);
    }

    fn shape(index: usize, candidates: usize, name: []const u8, rows: usize, width: usize, fp32: bool) !void {
        var buf: [64]u8 = undefined;
        var expected: []const u8 = "tower_decode0.candidate_logits";
        var expected_width = candidates;
        var expected_fp32 = true;
        if (index > 0) {
            const j = index - 1;
            const layer = if (j < C.shared_layers) j else C.shared_layers + (j - C.shared_layers) / 2;
            if (layer >= C.tower_layers) return error.UnexpectedProbe;
            const head = j >= C.shared_layers and (j - C.shared_layers) % 2 == 1;
            expected = if (head)
                try std.fmt.bufPrint(&buf, "tower_decode0.head{d}_logits", .{layer - C.shared_layers})
            else
                try std.fmt.bufPrint(&buf, "tower_decode0.layer{d:0>2}", .{layer});
            expected_width = if (head) C.codebook_size else C.hidden;
            expected_fp32 = head;
        }
        if (!std.mem.eql(u8, name, expected) or rows != 1 or width != expected_width or fp32 != expected_fp32)
            return error.ProbeSequenceOrShapeMismatch;
    }

    fn capture(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, fp32: bool) !void {
        const p: *LogitProbe = @ptrCast(@alignCast(ctx));
        try shape(p.next, p.engine.c.candidates(), name, rows, width, fp32);
        const bytes = try p.engine.gpa.alloc(u8, rows * width * @as(usize, if (fp32) 4 else 2));
        defer p.engine.gpa.free(bytes);
        // Download now: the next tower layer or head reuses the same device scratch.
        try p.engine.download(ptr, bytes);
        if (p.reference) |ref| {
            const expected = ref.saved[p.next] orelse return error.MissingReferenceProbe;
            if (expected.len != bytes.len) return error.ProbeSizeMismatch;
            if (firstDifference(expected, bytes)) |i| {
                std.debug.print("MISMATCH forced-logits probe={s} index={d} byte={d} reference={x} candidate={x}\n", .{ name, p.next, i, expected[i], bytes[i] });
                return error.ForcedLogitMismatch;
            }
        } else p.saved[p.next] = try p.engine.gpa.dupe(u8, bytes);
        p.next += 1;
        p.bytes += bytes.len;
    }

    fn complete(p: *const LogitProbe) !void {
        if (p.next != total or p.bytes == 0) return error.IncompleteProbeCoverage;
    }
};

test "live byte comparator probe coverage rejects missing names and incorrect shapes" {
    try LogitProbe.shape(0, 4097, "tower_decode0.candidate_logits", 1, 4097, true);
    try LogitProbe.shape(5, 4097, "tower_decode0.layer04", 1, C.hidden, false);
    try LogitProbe.shape(6, 4097, "tower_decode0.head0_logits", 1, C.codebook_size, true);
    try LogitProbe.shape(22, 4097, "tower_decode0.head8_logits", 1, C.codebook_size, true);
    try std.testing.expectError(error.UnexpectedProbe, LogitProbe.shape(23, 4097, "extra", 1, 1, true));
    try std.testing.expectError(error.ProbeSequenceOrShapeMismatch, LogitProbe.shape(0, 4097, "wrong", 1, 4097, true));
    try std.testing.expectError(error.ProbeSequenceOrShapeMismatch, LogitProbe.shape(0, 4097, "tower_decode0.candidate_logits", 0, 4097, true));
    try std.testing.expectError(error.ProbeSequenceOrShapeMismatch, LogitProbe.shape(6, 4097, "tower_decode0.head0_logits", 1, C.codebook_size, false));
}

fn poisonLiveKv(candidate: *E, rows: usize) !void {
    const stride = C.kv_heads * C.head_dim * 2;
    const bytes = try candidate.gpa.alloc(u8, rows * stride);
    defer candidate.gpa.free(bytes);
    @memset(bytes, 0xa5);
    for (0..C.layers) |li| {
        if (!candidate.c.full_attention[li]) continue;
        for (0..2) |kv| try candidate.put(candidate.kv_bb[li][kv], bytes);
    }
    for (0..C.tower_layers) |li| {
        for (0..2) |kv| try candidate.put(candidate.kv_tw[li][kv], bytes[0 .. (rows - 1) * stride]);
    }
    try candidate.flush();
}

fn forcedSeam(compare: *Compare, prefix: []const [W]i32, forced: []const [W]i32, chunk: usize, poison_kv: bool) !void {
    const base = compare.a;
    const candidate = compare.b;
    const joined = try base.gpa.alloc([W]i32, prefix.len + forced.len);
    defer base.gpa.free(joined);
    @memcpy(joined[0..prefix.len], prefix);
    @memcpy(joined[prefix.len..], forced);
    try d.model.validate(joined, base.c.ids, base.o.max_ctx, 1);
    for (forced) |row| {
        if (row[0] < base.c.ids.sem_begin or row[0] > base.c.ids.sem_end) return error.ForcedAcousticRowRequired;
    }
    const frame = try base.scratch(W * 4);
    const picks = try base.scratch(W * 4);
    for (forced) |row| {
        try base.put(frame, std.mem.asBytes(&row));
        try base.frameForced(.{ .frame = frame, .picks = picks }, null);
        try base.flush();
    }
    // Full prompt prefill writes tower rows through joined.len - 2, matching prefix tower rows plus forced steps.
    if (poison_kv) try poisonLiveKv(candidate, joined.len);
    try candidate.startPrompt(joined);
    var at: usize = 0;
    while (at < joined.len) {
        const count = @min(chunk, joined.len - at);
        try candidate.promptChunk(joined.len, at, count, null);
        at += count;
    }
    try candidate.finishPrompt(joined);
    compare.seam = joined.len;
    try compare.live(joined.len);
    try compare.range("hlast", 0, base.hlast, candidate.hlast, C.hidden * 2);
    var reference: LogitProbe = .{ .engine = base };
    defer reference.deinit();
    var observed: LogitProbe = .{ .engine = candidate, .reference = &reference };
    defer observed.deinit();
    const next_row = forced[forced.len - 1];
    try base.put(frame, std.mem.asBytes(&next_row));
    try base.frameForced(.{ .frame = frame, .picks = picks }, .{ .ctx = &reference, .f = LogitProbe.capture });
    try base.flush();
    try reference.complete();
    try candidate.frameForced(.{ .frame = frame, .picks = picks }, .{ .ctx = &observed, .f = LogitProbe.capture });
    try candidate.flush();
    try observed.complete();
    compare.seam = joined.len + 1;
    try compare.live(joined.len + 1);
    try compare.range("poststep.hlast", 0, base.hlast, candidate.hlast, C.hidden * 2);
    try compare.range("poststep.frame", 0, base.frames_out + forced.len * W * 4, candidate.frames_out, W * 4);
    var emitted: [W]i32 = undefined;
    try base.download(base.frames_out + forced.len * W * 4, std.mem.asBytes(&emitted));
    if (!std.mem.eql(i32, &emitted, &next_row)) return error.ForcedOutputMismatch;
    std.debug.print("PASS forced-seam prefix_rows={d} forced_rows={d} end_rows={d} buffers={d} bytes={d} next_step_probes={d} probe_bytes={d} poison_kv={}; no sampler, CFG, audio or independent fidelity qualification\n", .{ prefix.len, forced.len, joined.len, compare.buffers, compare.bytes, observed.next, observed.bytes, poison_kv });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    var model: ?[]const u8 = null;
    var path: ?[]const u8 = null;
    var forced_path: ?[]const u8 = null;
    var other_path: ?[]const u8 = null;
    var other_forced_path: ?[]const u8 = null;
    var poison_kv = false;
    var resume_cut: ?usize = null;
    var chunk: usize = 0;
    var context: usize = 8192;
    var i: usize = 1;
    while (i < argv.len) : (i += 2) {
        if (i + 1 >= argv.len) return error.MissingArgument;
        const key = argv[i];
        const value = argv[i + 1];
        if (std.mem.eql(u8, key, "--model")) {
            model = value;
        } else if (std.mem.eql(u8, key, "--rows")) {
            path = value;
        } else if (std.mem.eql(u8, key, "--forced-rows")) {
            forced_path = value;
        } else if (std.mem.eql(u8, key, "--other-rows")) {
            other_path = value;
        } else if (std.mem.eql(u8, key, "--other-forced-rows")) {
            other_forced_path = value;
        } else if (std.mem.eql(u8, key, "--poison-kv")) {
            if (!std.mem.eql(u8, value, "1")) return error.BadPoisonKv;
            poison_kv = true;
        } else if (std.mem.eql(u8, key, "--resume-cut")) {
            resume_cut = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "--chunk")) {
            chunk = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "--context")) {
            context = try std.fmt.parseInt(usize, value, 10);
        } else return error.UnknownArgument;
    }
    if ((other_path == null) != (other_forced_path == null) or (other_path != null and forced_path == null)) return error.IsolationRequiresBothHistories;
    if (poison_kv and forced_path == null) return error.PoisonKvRequiresForcedRows;
    if (model == null or path == null) return error.ModelAndRowsRequired;
    if (context < 1025 or context > 8192) return error.BadContext;
    if (other_path != null) try isolation.validateChunk(chunk);
    var opts: d.Options = .{ .max_ctx = context, .max_rows = chunk, .attn_chunk = @min(chunk, 1024), .max_frames = 1, .warmup = false };
    try opts.validatePrefill();
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, path.?, a, .limited(8192 * W * 4));
    if (raw.len % (W * 4) != 0) return error.BadRows;
    const n = raw.len / (W * 4);
    if (n < 2 or n > context - 2) return error.BadPromptWindow;
    if (resume_cut) |cut| try validateResumeCut(n, cut, chunk);
    const rows = try a.alloc([W]i32, n);
    @memcpy(std.mem.sliceAsBytes(rows), raw);
    var forced: []const [W]i32 = &.{};
    if (forced_path) |p| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, p, a, .limited(8192 * W * 4));
        if (bytes.len % (W * 4) != 0) return error.BadForcedRows;
        const count = bytes.len / (W * 4);
        _ = try forcedWindow(n, count, context);
        const storage = try a.alloc([W]i32, count);
        @memcpy(std.mem.sliceAsBytes(storage), bytes);
        forced = storage;
        opts.max_frames = count + 1;
    }
    const config = try C.Config.load(a, init.io, model.?);
    try d.model.validate(rows, config.ids, context, @max(forced.len + 1, 1));
    if (forced.len > 0) {
        const joined = try a.alloc([W]i32, rows.len + forced.len);
        @memcpy(joined[0..rows.len], rows);
        @memcpy(joined[rows.len..], forced);
        try d.model.validate(joined, config.ids, context, 1);
        for (forced) |row| if (row[0] < config.ids.sem_begin or row[0] > config.ids.sem_end)
            return error.ForcedAcousticRowRequired;
    }
    var other: ?isolation.History = null;
    if (other_path) |p| {
        other = .{ .rows = try isolation.readRows(a, init.io, p), .forced = try isolation.readRows(a, init.io, other_forced_path.?) };
        const first: isolation.History = .{ .rows = rows, .forced = forced };
        try isolation.validate(a, first, config.ids, context);
        try isolation.validate(a, other.?, config.ids, context);
        try isolation.distinct(first, other.?);
        opts.max_frames = @max(forced.len, other.?.forced.len) + 1;
        try d.load_limit.admitStates(a, init.io, model.?, @intCast(context), 4);
    }
    const base = try E.create(std.heap.page_allocator, init.io, model.?, opts);
    defer base.destroy();
    const candidate = try base.fork();
    defer candidate.destroy();
    try base.startPrompt(rows);
    try candidate.startPrompt(rows);
    var compare: Compare = .{ .a = base, .b = candidate };
    var at: usize = 0;
    while (at < n) {
        const count = @min(chunk, n - at);
        for (at..at + count) |r| try base.promptChunk(n, r, 1, null);
        try base.flush();
        try candidate.promptChunk(n, at, count, null);
        try candidate.flush();
        at += count;
        compare.seam = at;
        try compare.live(n);
    }
    try base.finishPrompt(rows);
    try candidate.finishPrompt(rows);
    try compare.final();
    if (resume_cut) |cut| try resumedState(base, candidate, rows, cut, chunk);
    if (forced.len > 0) {
        var seam: Compare = .{ .a = base, .b = candidate };
        try forcedSeam(&seam, rows, forced, chunk, poison_kv);
    }
    if (other) |second| try isolation.run(Compare, base, candidate, .{ .rows = rows, .forced = forced }, second, chunk);
    std.debug.print("PASS chunk-state rows={d} context={d} reference_chunk=1 candidate_chunk={d} buffers={d} bytes={d}; no decode, audio or independent fidelity qualification\n", .{ n, context, chunk, compare.buffers, compare.bytes });
}
