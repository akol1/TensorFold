//! A row's full-vocabulary scores without devices: the stats layout the kernel writes and the labels a block reads.
const std = @import("std");
const cuda = @import("cuda");
const K = cuda.qwen35_scores.Kernels;
const t = std.testing;

test "score stats: a row's two words and its sixteen logits share no word" {
    // `stats` holds log-sum-exp and argmax from 0, two words a row, and the selected logits from 8, sixteen a row.
    for ([_]usize{ 1, 2, 3, 4 }) |rows| {
        try t.expect(rows * 2 <= cuda.qwen35_scores.stats_base);
        try t.expectEqual(cuda.qwen35_scores.stats_base + rows * cuda.qwen35_scores.stats_width, K.statsWords(rows));
    }
    // A caller must not assume a run's head clears the logits: at 128 rows the two words a row pass 8.
    try t.expectEqual(@as(usize, 8), cuda.qwen35_scores.stats_base);
    try t.expectEqual(@as(usize, 16), cuda.qwen35_scores.stats_width);
}

test "score stats: a label row is a count and the ids the kernel reads beside it" {
    // The kernel reads labels[r * 17] as a count and labels[r * 17 + 1 ..][0..16] as ids, so a row spans 17 words.
    try t.expectEqual(@as(usize, 17), cuda.qwen35_scores.labels_width);
    try t.expect(cuda.qwen35_scores.labels_width == 1 + cuda.qwen35_scores.stats_width);
    // Four rows, the most a batched request seeds, stay clear of each other.
    for (0..4) |r| {
        try t.expectEqual(r * 17 + 17, (r + 1) * 17);
    }
}

test "score stats: the grid is one block a row so a row never sees its batch's shape" {
    // One block a row with 256 threads: the row's reduction belongs to its own block, not to the launch.
    try t.expectEqual(@as(u32, 256), cuda.qwen35_scores.threads);
    // A single row's stats words are exactly the first sixteen logits after the two-word head.
    try t.expectEqual(@as(usize, 8 + 16), K.statsWords(1));
}
