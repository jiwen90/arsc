ar_r <- "E:/autoresearch/tools/R/ar.R"; if (file.exists(ar_r)) source(ar_r)  # autoresearch toolkit, when vendored there
# Rebuild the R harmony references in $ARSC_REF_HARMONY for every harmony release arsc reproduces (1.2.0,
# 2.0.5, 2.1.0): runs benchmarks/r_reference_harmony.R once per release, each in its own R process with that
# release's library first on .libPaths (one R session cannot load two harmony versions), then checks that each
# output was made by the expected version.
#
# Usage: Rscript benchmarks/r_reference_harmony_all.R [version ...]     (default: all three, one after another)
#
# Libraries (each holds only harmony; the rest comes from the normal library):
#   1.2.0  the default library (ARSC_HARMONY_LIB_1_2_0 to override)
#   2.0.5  ARSC_HARMONY_LIB_2_0_5, default <AR_HOME>/.Rlib_harmony_2.0.5
#   2.1.0  ARSC_HARMONY_LIB_2_1_0, default <AR_HOME>/.Rlib_harmony_2.1.0
# AR_HOME is the autoresearch checkout (set by ar.R); elsewhere set the variables above.
# Output folder: ARSC_REF_HARMONY; when unset, <AR_HOME>/.cache/arsc/ref_harmony under autoresearch (a folder its
# scratch cleanup never prunes), else $ARSC_SCRATCH/ref_harmony. The input (<folder>/input) comes from
# benchmarks/harmony_input.py; rerun that first only to change the dataset. About 9 min per version (38,440 cells,
# 5 cases, 2 seeds); versions can run in parallel as separate processes (one version each).
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
home <- if (exists("AR_HOME")) AR_HOME else Sys.getenv("AR_HOME", normalizePath(file.path(here, "..", "..", ".."),
                                                                                 mustWork = FALSE))
lib_for <- function(v) {
  env <- Sys.getenv(paste0("ARSC_HARMONY_LIB_", gsub(".", "_", v, fixed = TRUE)))
  if (nzchar(env)) return(env)
  if (v == "1.2.0") "" else file.path(home, paste0(".Rlib_harmony_", v))
}
versions <- commandArgs(trailingOnly = TRUE)
if (!length(versions)) versions <- c("1.2.0", "2.0.5", "2.1.0")
rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
refs <- Sys.getenv("ARSC_REF_HARMONY")
if (!nzchar(refs)) refs <- if (exists("AR_HOME")) file.path(AR_HOME, ".cache", "arsc", "ref_harmony") else
  file.path(Sys.getenv("ARSC_SCRATCH", "scratch"), "ref_harmony")
if (!file.exists(file.path(refs, "input", "pcs.json"))) stop("no input in ", refs, "/input: run benchmarks/harmony_input.py")
for (v in versions) {
  lib <- lib_for(v)
  if (nzchar(lib) && !dir.exists(file.path(lib, "harmony"))) stop("no harmony in ", lib, " (for ", v, ")")
  args <- c(file.path(here, "r_reference_harmony.R"), file.path(refs, "input"), refs, if (nzchar(lib)) lib)
  unlink(file.path(refs, v), recursive = TRUE)  # so a run that loads another version cannot pass on old files
  cat("== harmony", v, if (nzchar(lib)) paste("from", lib) else "from the default library", "\n")
  st <- system2(rscript, shQuote(args))
  if (st != 0) stop("r_reference_harmony.R failed for harmony ", v)
  tr <- jsonlite::fromJSON(file.path(refs, v, "lane1_trace.json"))
  if (!identical(tr$harmony, v)) stop("expected harmony ", v, ", the output says ", tr$harmony)
}
cat("references in", refs, "\n")
