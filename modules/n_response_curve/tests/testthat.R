library(testthat)

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) {
  stop("testthat.R must be executed with Rscript")
}
script_path <- normalizePath(sub("^--file=", "", script_argument), winslash = "/", mustWork = TRUE)
test_dir(file.path(dirname(script_path), "testthat"), reporter = "summary")
