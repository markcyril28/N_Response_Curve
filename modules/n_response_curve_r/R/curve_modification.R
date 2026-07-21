nrc_run_curve_modification <- function(stage) {
  specification <- stage$contract$specification
  model_formula <- nrc_model_formula(specification, stage$data)
  if (is.null(model_formula)) {
    return(nrc_skip_result("MODEL_SPECIFICATION_REQUIRED"))
  }
  formula_text <- paste(deparse(model_formula), collapse = " ")
  n_rate_column <- specification$n_rate_column
  if (is.null(n_rate_column)) {
    n_rate_column <- "n_rate_kg_ha"
  }
  if (!n_rate_column %in% all.vars(model_formula)) {
    return(nrc_skip_result("N_RATE_TERM_REQUIRED"))
  }
  if (isTRUE(specification$require_quadratic_n) &&
        !grepl(paste0("I(", n_rate_column, "^2)"), formula_text, fixed = TRUE)) {
    return(nrc_skip_result("QUADRATIC_N_TERM_REQUIRED"))
  }
  nrc_run_mixed_models(stage)
}
