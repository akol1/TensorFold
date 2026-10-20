//! DeepSeek V4.1's sparse attention pipelines: declared geometry, M1 and M2 limits, the parallel launch's reservation.
const limits = @import("limits.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "deepseek_v41 sparse dsv41_sparse_attention", .threads = 32, .tg_memory = 256, .m1 = 768, .m2 = 768 },
    .{ .name = "deepseek_v41 sparse dsv41_sparse_attention_parallel", .threads = 256, .tg_memory = 8192, .reserved = 256 },
};
