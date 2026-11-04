# SALSO estimate (VI loss) for each label matrix listed in <dir>/manifest.csv,
# with the number of tiers capped at `cap`.
suppressPackageStartupMessages(library(salso))
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
seed <- if (length(args) >= 2) as.integer(args[2]) else 2026L
manifest <- read.csv(file.path(dir, "manifest.csv"), stringsAsFactors = FALSE)
for (i in seq_len(nrow(manifest))) {
  draws <- as.matrix(read.csv(file.path(dir, manifest$file[i]), header = FALSE))
  set.seed(seed)
  est <- salso(draws, loss = VI(), nRuns = 64, maxNClusters = manifest$cap[i], nCores = 1)
  write.csv(data.frame(tier = as.integer(est)), file.path(dir, paste0("est_", manifest$file[i])), row.names = FALSE)
}
