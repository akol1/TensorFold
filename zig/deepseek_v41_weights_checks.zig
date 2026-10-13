//! Host checks for V4.1 configuration, Engram table plans and resident weight staging; no kernels compiled.
test {
    _ = @import("src/families/deepseek_v41/metal/config.zig");
    _ = @import("src/families/deepseek_v41/metal/view.zig");
    _ = @import("src/families/deepseek_v41/engram_hash.zig");
    _ = @import("src/families/deepseek_v41/metal/load.zig");
    _ = @import("src/families/deepseek_v41/metal/weights.zig");
}
