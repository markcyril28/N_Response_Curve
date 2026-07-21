nrc_run_marginal_contrasts <- function(stage) {
  specification <- stage$contract$specification
  contrast_specification <- specification$contrast_specification
  if (is.null(contrast_specification) || !is.list(contrast_specification)) {
    return(nrc_skip_result("CONTRAST_SPECIFICATION_REQUIRED"))
  }
  nrc_skip_result("CONTRAST_MODEL_RESULT_REQUIRED")
}
