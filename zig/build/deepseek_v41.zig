//! Experimental DS4.1 operator and independent-reference qualification targets.
const std = @import("std");

pub fn checks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    metal: *std.Build.Module,
    sources: *std.Build.Module,
    engine: *std.Build.Module,
) void {
    const device = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_checks.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = sources },
        },
    }) });
    b.step("test-deepseek-v41", "Experimental DS4.1 EXL3, Engram, mHC and sparse attention on Metal")
        .dependOn(&b.addRunArtifact(device).step);
    const reference = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/deepseek_v41_reference_checks.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "tensorfold", .module = engine },
                .{ .name = "metal", .module = metal },
            },
        }),
        .filters = &.{"tests.deepseek_v41_reference."},
    });
    b.step("test-deepseek-v41-reference", "Independent CPU teacher-oracle arithmetic boundaries")
        .dependOn(&b.addRunArtifact(reference).step);
}
