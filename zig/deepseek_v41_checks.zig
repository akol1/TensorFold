//! Source-built V4.1 operator qualification; real Metal execution, no checkpoint needed.
test {
    _ = @import("src/core/exl3_projection_gpu_test.zig");
    _ = @import("src/families/deepseek_v41/metal/engram.zig");
    _ = @import("src/families/deepseek_v41/metal/csa2.zig");
    _ = @import("src/families/deepseek_v41/metal/ops_check.zig");
    _ = @import("src/families/deepseek_v41/metal/load.zig");
    _ = @import("src/families/deepseek_v41/metal/index_precision_test.zig");
    _ = @import("src/families/deepseek_v41/metal/index_select_test.zig");
    _ = @import("src/families/deepseek_v41/metal/config.zig");
    _ = @import("src/families/deepseek_v41/metal/profiler.zig");
    _ = @import("src/core/gpu_counters.zig");
    _ = @import("src/core/expert_groups.zig");
}
