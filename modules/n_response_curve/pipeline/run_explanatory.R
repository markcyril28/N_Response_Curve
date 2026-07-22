arguments <- commandArgs(trailingOnly = TRUE)
script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) {
  stop("Unable to resolve the R entry point path", call. = FALSE)
}
script_path <- normalizePath(sub("^--file=", "", script_argument), winslash = "/", mustWork = TRUE)
analysis_stage_root <- normalizePath(
  file.path(dirname(script_path), "..", "analysis", "stages"),
  winslash = "/",
  mustWork = TRUE
)
source(file.path(analysis_stage_root, "contracts.R"))
source(file.path(analysis_stage_root, "diagnostics.R"))
source(file.path(analysis_stage_root, "mixed_models.R"))
source(file.path(analysis_stage_root, "curve_modification.R"))
source(file.path(analysis_stage_root, "marginal_contrasts.R"))

nrc_dispatch_analysis <- function(stage) {
  specification <- stage$contract$specification
  analysis_family <- specification$analysis_family
  if (is.null(analysis_family) || !is.character(analysis_family) || length(analysis_family) != 1L) {
    return(nrc_failed_result("ANALYSIS_FAMILY_REQUIRED", "R specification lacks analysis_family"))
  }
  if (identical(analysis_family, "observation_level_curve_modification")) {
    return(nrc_run_curve_modification(stage))
  }
  if (identical(analysis_family, "marginal_contrasts")) {
    return(nrc_run_marginal_contrasts(stage))
  }
  if (analysis_family %in% c(
    "one_factor_inferential",
    "all_supported_interactions",
    "multivariable_mixed_effects"
  )) {
    return(nrc_run_mixed_models(stage))
  }
  nrc_skip_result("R_ANALYSIS_FAMILY_NOT_IMPLEMENTED")
}

contract_path <- nrc_parse_contract_path(arguments)
stage <- nrc_read_contract(contract_path)
result <- nrc_dispatch_analysis(stage)
nrc_write_stage_result(stage, result)
