//! V4.1's resident model behind the existing serialized request worker.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const Serial = api.serial_host;
const Allocator = std.mem.Allocator;

pub const Host = struct {
    gpa: Allocator,
    model: *ds.Model,
    serial: *Serial.Host,
    sampler_storage: []u8,
    memory_info: api.Memory,
    warm: mtl.keepalive.Target,

    fn self(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        try self(ptr).model.reset();
    }
    fn chunk(ptr: *anyopaque, ids: []const u32, last: bool) !void {
        _ = try self(ptr).model.forward(ids, if (last) .last else .none);
    }
    fn advance(ptr: *anyopaque, token: u32) !void {
        _ = try self(ptr).model.forward(&.{token}, .last);
    }
    fn fed(ptr: *anyopaque) u64 {
        return self(ptr).model.cache.position;
    }
    fn draw(ptr: *anyopaque, sampling: ?api.Sampling) !u32 {
        const h = self(ptr);
        if (!h.model.valid or h.model.logits_rows != 1) return error.RoundNotReady;
        const row = h.model.workspace.logits(1);
        if (sampling) |s| if (s.temperature > 0) {
            for (row) |value| if (!std.math.isFinite(value)) return error.NonfiniteLogits;
            var arena = std.heap.FixedBufferAllocator.init(h.sampler_storage);
            return Serial.choose(arena.allocator(), .{ .f32 = row }, h.model.cache.position, s);
        };
        var best: u32 = 0;
        var maximum = -std.math.inf(f32);
        for (row, 0..) |value, id| {
            if (!std.math.isFinite(value)) return error.NonfiniteLogits;
            if (value > maximum) {
                maximum = value;
                best = @intCast(id);
            }
        }
        return best;
    }
    fn memory(ptr: ?*anyopaque, _: bool) ?api.Memory {
        const h: *Host = @ptrCast(@alignCast(ptr.?));
        return h.memory_info;
    }
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.speed_up != null or o.slide or o.learn != null or
        (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or o.prompt_cache_over_cap)
    {
        problem.* = "V4.1 resident serving does not support speed-up, Sliding Weights or retained prompt-cache flags";
        return null;
    }
    if (o.drafter != null or o.device != null or o.segments != null or
        o.mtp_drafts != null or o.mtp_confidence != null or
        (o.keep != null and o.keep.? != 0) or o.tp != 1 or o.rank != 0 or
        o.master.len != 0 or o.policy.len != 0)
    {
        problem.* = "experimental V4.1 serving does not support drafter, MTP, cache slots, device, TP or policy flags";
        return null;
    }
    if (o.lanes_fixed and o.lanes != 1) {
        problem.* = "V4.1's plain resident worker serves one active sequence; use --parallel 1 or auto";
        return null;
    }
    const requested = o.context orelse 0;
    if ((o.context != null and requested <= 0) or requested > 1048576) {
        problem.* = "DeepSeek V4.1 context must be from 1 to 1,048,576 tokens (omitted: full model limit)";
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    const model = try gpa.create(ds.Model);
    errdefer gpa.destroy(model);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const token_map = try std.fs.path.join(a, &.{ o.dir, "engram-token-map.bin" });
    model.* = ds.Model.open(gpa, io, .{ .dir = o.dir, .table_dir = o.engram_table_dir, .token_map = token_map, .context = @intCast(requested) }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the full-resident DeepSeek V4.1 engine cannot load ({s})", .{@errorName(err)});
        gpa.destroy(model);
        gpa.destroy(h);
        return null;
    };
    errdefer model.deinit();
    const sampler_storage = try gpa.alloc(u8, 8 << 20);
    errdefer gpa.free(sampler_storage);
    const bytes = model.admitted_bytes;
    h.* = .{ .gpa = gpa, .model = model, .serial = undefined, .sampler_storage = sampler_storage, .memory_info = .{ .active = bytes, .peak = bytes, .pinned = model.residency.allocatedSize() }, .warm = .{ .queue = model.queue, .sets = @as(*const [1]mtl.ResidencySet, &model.residency) } };
    const startup = try std.fmt.allocPrint(a, "EXPERIMENTAL / UNQUALIFIED DeepSeek V4.1: whole-model accuracy gate has not passed; " ++
        "all weights resident ({d} bytes), packed caches + bounded scratch ({d} bytes), context {d}; no SSD lookups", .{ model.checkpoint.residentBytes(), bytes - model.checkpoint.residentBytes(), model.cache.plan.capacity });
    h.serial = try Serial.Host.init(gpa, io, .{
        .ctx = h,
        .info = .{ .name = "deepseek-v41-metal-exl3-resident", .lanes = 1, .context_window = model.cache.plan.capacity, .prefill_step = model.workspace.plan.rows, .plain_only = true, .startup = startup },
        .reset = Host.reset,
        .prompt_chunk = Host.chunk,
        .decode_chunk = Host.chunk,
        .draw = Host.draw,
        .advance = Host.advance,
        .fed = Host.fed,
        .memory = .{ .ctx = h, .read = Host.memory },
        .keepalive = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick },
    });
    return .{ .engine = h.serial.engine(), .close = close, .ctx = h };
}

pub fn close(ptr: *anyopaque) void {
    const h = Host.self(ptr);
    h.serial.deinit();
    h.model.deinit();
    h.gpa.destroy(h.model);
    h.gpa.free(h.sampler_storage);
    h.gpa.destroy(h);
}

test "unsupported DeepSeek serving flags are refused before allocating or loading a checkpoint" {
    var problem: []const u8 = "";
    const bad_options = [_]api.Open{
        .{ .dir = "", .model_type = "deepseek_v41", .tp = 2 },
        .{ .dir = "", .model_type = "deepseek_v41", .policy = "unknown=1" },
        .{ .dir = "", .model_type = "deepseek_v41", .drafter = "draft" },
        .{ .dir = "", .model_type = "deepseek_v41", .mtp_drafts = 1 },
        .{ .dir = "", .model_type = "deepseek_v41", .device = 0 },
        .{ .dir = "", .model_type = "deepseek_v41", .keep = 1 },
    };
    for (bad_options) |options| {
        try std.testing.expect(try open(std.testing.failing_allocator, std.testing.failing_allocator, std.testing.io, options, &problem) == null);
    }
}
