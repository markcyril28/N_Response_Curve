args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  message("Expected input CSV and output JSON paths")
  quit(status = 2L)
}

input_path <- args[[1L]]
output_path <- args[[2L]]

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
})

write_result <- function(result) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(
    jsonlite::toJSON(
      result,
      auto_unbox = TRUE,
      pretty = TRUE,
      digits = 15,
      null = "null",
      na = "null"
    ),
    output_path,
    useBytes = TRUE
  )
}

fail <- function(reason_code, message_text) {
  write_result(list(
    status = "failed",
    reason_code = reason_code,
    error_message = message_text,
    engine = "R/lme4",
    analysis_role = "exploratory_heterogeneity_diagnostic_not_causal"
  ))
  message(message_text)
  quit(status = 2L)
}

if (!file.exists(input_path)) {
  fail("INPUT_MISSING", paste("Input CSV does not exist:", input_path))
}

data <- tryCatch(
  utils::read.csv(
    input_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("", "NA", "NaN")
  ),
  error = function(error) error
)
if (inherits(data, "error")) {
  fail("INPUT_PARSE_FAILED", conditionMessage(data))
}

required <- c(
  "release_record_uid",
  "response_series_uid",
  "study_uid",
  "trial_uid",
  "n_rate_kg_ha",
  "yield_t_ha"
)
missing_columns <- setdiff(required, names(data))
if (length(missing_columns)) {
  fail(
    "INPUT_COLUMNS_MISSING",
    paste("Input CSV is missing columns:", paste(missing_columns, collapse = ", "))
  )
}
if (anyNA(data$release_record_uid) || anyDuplicated(data$release_record_uid)) {
  fail("INPUT_RECORD_UID_INVALID", "release_record_uid must be unique and nonmissing")
}
for (key in c("release_record_uid", "response_series_uid", "study_uid", "trial_uid")) {
  if (anyNA(data[[key]]) || any(!nzchar(trimws(as.character(data[[key]]))))) {
    fail("INPUT_GROUP_UID_INVALID", paste(key, "must be nonmissing and nonblank"))
  }
}

data$n_rate_kg_ha <- suppressWarnings(as.numeric(data$n_rate_kg_ha))
data$yield_t_ha <- suppressWarnings(as.numeric(data$yield_t_ha))
if (any(!is.finite(data$n_rate_kg_ha)) || any(!is.finite(data$yield_t_ha))) {
  fail("INPUT_NONFINITE", "N rate and yield must be finite")
}
if (nrow(data) < 12L) {
  fail("INSUFFICIENT_OBSERVATIONS", "At least 12 observations are required")
}
series_counts <- table(data$response_series_uid)
if (length(series_counts) < 4L || any(series_counts < 3L)) {
  fail(
    "INSUFFICIENT_SERIES_SUPPORT",
    "At least four series with at least three observations each are required"
  )
}
distinct_levels <- vapply(
  split(data$n_rate_kg_ha, data$response_series_uid),
  function(values) length(unique(values)),
  integer(1)
)
if (any(distinct_levels < 3L)) {
  fail("INSUFFICIENT_N_LEVELS", "Every series must contain at least three N levels")
}

data$response_series_uid <- factor(data$response_series_uid)
data$n_rate_per_100_kg_ha <- data$n_rate_kg_ha / 100
warning_messages <- character()
model <- tryCatch(
  withCallingHandlers(
    lme4::lmer(
      yield_t_ha ~ n_rate_per_100_kg_ha +
        (1 + n_rate_per_100_kg_ha | response_series_uid),
      data = data,
      REML = FALSE,
      control = lme4::lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 100000),
        check.conv.singular = "ignore"
      )
    ),
    warning = function(warning) {
      warning_messages <<- c(warning_messages, conditionMessage(warning))
      invokeRestart("muffleWarning")
    }
  ),
  error = function(error) error
)
if (inherits(model, "error")) {
  fail("MIXED_MODEL_FIT_FAILED", conditionMessage(model))
}

convergence_messages <- model@optinfo$conv$lme4$messages
optimizer_code <- model@optinfo$conv$opt
converged <- is.null(convergence_messages) &&
  (is.null(optimizer_code) || all(optimizer_code == 0L))
if (!converged) {
  fail(
    "MIXED_MODEL_NONCONVERGENCE",
    paste(c(as.character(convergence_messages), paste("optimizer code:", optimizer_code)), collapse = "; ")
  )
}

fixed <- lme4::fixef(model)
fixed_covariance <- as.matrix(stats::vcov(model))
fixed_se <- sqrt(diag(fixed_covariance))
random_covariance <- as.matrix(lme4::VarCorr(model)$response_series_uid)
random_sd <- attr(lme4::VarCorr(model)$response_series_uid, "stddev")
random_correlation <- attr(lme4::VarCorr(model)$response_series_uid, "correlation")

result <- list(
  status = "completed",
  engine = "R/lme4",
  model = "random_intercept_random_slope_by_response_series",
  formula = paste(
    "yield_t_ha ~ n_rate_per_100_kg_ha +",
    "(1 + n_rate_per_100_kg_ha | response_series_uid)"
  ),
  analysis_role = "exploratory_heterogeneity_diagnostic_not_causal",
  observations = nrow(data),
  study_count = length(unique(data$study_uid)),
  trial_count = length(unique(data$trial_uid)),
  series_count = length(levels(data$response_series_uid)),
  converged = TRUE,
  singular = isTRUE(lme4::isSingular(model, tol = 1e-4)),
  warning_messages = as.list(unique(warning_messages)),
  fixed_intercept_t_ha = unname(fixed[["(Intercept)"]]),
  fixed_intercept_se_t_ha = unname(fixed_se[["(Intercept)"]]),
  fixed_slope_t_ha_per_100_kg_n_ha = unname(fixed[["n_rate_per_100_kg_ha"]]),
  fixed_slope_se_t_ha_per_100_kg_n_ha = unname(fixed_se[["n_rate_per_100_kg_ha"]]),
  random_intercept_sd_t_ha = unname(random_sd[[1L]]),
  random_slope_sd_t_ha_per_100_kg_n_ha = unname(random_sd[[2L]]),
  random_intercept_slope_correlation = unname(random_correlation[1L, 2L]),
  random_effect_covariance = unname(random_covariance[1L, 2L]),
  residual_sd_t_ha = stats::sigma(model),
  log_likelihood = as.numeric(stats::logLik(model)),
  aic = stats::AIC(model),
  bic = stats::BIC(model)
)
write_result(result)
quit(status = 0L)
