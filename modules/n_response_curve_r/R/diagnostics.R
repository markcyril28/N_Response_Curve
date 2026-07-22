nrc_package_versions <- function() {
  packages <- c("arrow", "broom", "digest", "emmeans", "glmmTMB", "jsonlite", "lme4", "nnet", "TMB")
  installed <- utils::installed.packages()[, "Version"]
  versions <- lapply(packages, function(package_name) {
    if (!package_name %in% names(installed)) {
      return(NA_character_)
    }
    unname(installed[[package_name]])
  })
  stats::setNames(unlist(versions, use.names = FALSE), packages)
}

nrc_model_converged <- function(model) {
  if (inherits(model, "merMod")) {
    messages <- model@optinfo$conv$lme4$messages
    optimizer_code <- model@optinfo$conv$opt
    return(is.null(messages) && (is.null(optimizer_code) || identical(optimizer_code, 0L)))
  }
  if (inherits(model, "glmmTMB")) {
    return(isTRUE(model$fit$convergence == 0L) && isTRUE(model$sdr$pdHess))
  }
  if (inherits(model, "multinom")) {
    return(isTRUE(model$convergence == 0L))
  }
  isTRUE(model$converged) || inherits(model, "lm")
}

nrc_residual_summary <- function(model) {
  residual_values <- tryCatch(as.numeric(stats::residuals(model)), error = function(error) numeric())
  residual_values <- residual_values[is.finite(residual_values)]
  if (!length(residual_values)) {
    return(list(available = FALSE))
  }
  quantiles <- stats::quantile(residual_values, probs = c(0, 0.25, 0.5, 0.75, 1), names = FALSE)
  list(
    available = TRUE,
    minimum = unname(quantiles[[1L]]),
    first_quartile = unname(quantiles[[2L]]),
    median = unname(quantiles[[3L]]),
    third_quartile = unname(quantiles[[4L]]),
    maximum = unname(quantiles[[5L]])
  )
}

nrc_influence_summary <- function(model) {
  leverage <- tryCatch(as.numeric(stats::hatvalues(model)), error = function(error) numeric())
  cooks_distance <- tryCatch(as.numeric(stats::cooks.distance(model)), error = function(error) numeric())
  leverage <- leverage[is.finite(leverage)]
  cooks_distance <- cooks_distance[is.finite(cooks_distance)]
  list(
    available = length(leverage) > 0L || length(cooks_distance) > 0L,
    maximum_leverage = if (length(leverage)) max(leverage) else NA_real_,
    maximum_cooks_distance = if (length(cooks_distance)) max(cooks_distance) else NA_real_
  )
}

nrc_contrast_coding <- function(model) {
  coding <- tryCatch(attr(stats::model.matrix(model), "contrasts"), error = function(error) NULL)
  if (is.null(coding)) {
    return(list())
  }
  as.list(coding)
}

nrc_model_diagnostics <- function(model, warnings = character(), input_row_count = NA_integer_) {
  singular <- inherits(model, "merMod") && lme4::isSingular(model, tol = 1e-4)
  fitted_rows <- tryCatch(stats::nobs(model), error = function(error) NA_integer_)
  dropped_rows <- if (is.na(input_row_count) || is.na(fitted_rows)) {
    NA_integer_
  } else {
    as.integer(input_row_count - fitted_rows)
  }
  list(
    converged = nrc_model_converged(model),
    singular = singular,
    boundary_fit = singular,
    input_row_count = as.integer(input_row_count),
    fitted_row_count = as.integer(fitted_rows),
    dropped_row_count = dropped_rows,
    residual_df = tryCatch(stats::df.residual(model), error = function(error) NA_integer_),
    residual_summary = nrc_residual_summary(model),
    influence = nrc_influence_summary(model),
    warnings = as.list(unique(warnings)),
    contrast_coding = nrc_contrast_coding(model),
    package_versions = as.list(nrc_package_versions()),
    r_version = R.version.string
  )
}

nrc_tidy_model <- function(model) {
  if (!requireNamespace("broom", quietly = TRUE)) {
    return(list())
  }
  tidy <- tryCatch(
    broom::tidy(model, conf.int = TRUE),
    error = function(error) broom::tidy(model)
  )
  nrc_frame_to_rows(as.data.frame(tidy))
}

nrc_apply_multiplicity <- function(rows, specification) {
  multiplicity <- specification$multiplicity
  if (is.null(multiplicity)) {
    return(rows)
  }
  method <- multiplicity$method
  family_id <- multiplicity$family_id
  if (is.null(method)) {
    method <- "BH"
  }
  indexes <- which(vapply(
    rows,
    function(row) !is.null(row$p.value) && is.finite(as.numeric(row$p.value)),
    logical(1)
  ))
  if (!length(indexes)) {
    return(rows)
  }
  raw_values <- vapply(indexes, function(index) as.numeric(rows[[index]]$p.value), numeric(1))
  adjusted_values <- stats::p.adjust(raw_values, method = method)
  for (position in seq_along(indexes)) {
    index <- indexes[[position]]
    rows[[index]]$p.value_raw <- raw_values[[position]]
    rows[[index]]$p.value_adjusted <- adjusted_values[[position]]
    rows[[index]]$multiplicity_method <- method
    rows[[index]]$multiplicity_family_id <- family_id
  }
  rows
}
