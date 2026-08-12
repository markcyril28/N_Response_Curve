nrc_package_versions <- function() {
  packages <- c(
    "arrow", "broom", "broom.mixed", "digest", "emmeans", "glmmTMB",
    "jsonlite", "lme4", "nnet", "performance", "reformulas", "TMB"
  )
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
  singular <- if (inherits(model, "merMod")) {
    lme4::isSingular(model, tol = 1e-4)
  } else if (inherits(model, "glmmTMB") && requireNamespace("performance", quietly = TRUE)) {
    checked <- tryCatch(performance::check_singularity(model), error = function(error) FALSE)
    isTRUE(as.logical(checked)[[1L]])
  } else {
    FALSE
  }
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
  mixed <- inherits(model, "merMod") || inherits(model, "glmmTMB")
  if (mixed && !requireNamespace("broom.mixed", quietly = TRUE)) {
    return(list())
  }
  if (!mixed && !requireNamespace("broom", quietly = TRUE)) {
    return(list())
  }
  tidy <- tryCatch({
    if (mixed) {
      broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    } else {
      broom::tidy(model, conf.int = TRUE)
    }
  }, error = function(error) NULL)
  if (is.null(tidy) || !nrow(tidy)) {
    return(list())
  }
  nrc_frame_to_rows(as.data.frame(tidy))
}

nrc_fixed_effect_design_reason <- function(model_formula, data, minimum_residual_df = 3L) {
  fixed_formula <- tryCatch(
    if (requireNamespace("reformulas", quietly = TRUE)) {
      reformulas::nobars(model_formula)
    } else if (requireNamespace("lme4", quietly = TRUE)) {
      suppressWarnings(lme4::nobars(model_formula))
    } else {
      model_formula
    },
    error = function(error) model_formula
  )
  model_frame <- tryCatch(
    stats::model.frame(fixed_formula, data = data, na.action = stats::na.fail),
    error = function(error) error
  )
  if (inherits(model_frame, "error")) {
    return("FIXED_EFFECT_MODEL_FRAME_FAILED")
  }
  design <- tryCatch(stats::model.matrix(fixed_formula, data = model_frame), error = function(error) error)
  if (inherits(design, "error") || !is.matrix(design) || !ncol(design)) {
    return("FIXED_EFFECT_DESIGN_MATRIX_FAILED")
  }
  if (qr(design)$rank < ncol(design)) {
    return("ALIASED_DESIGN_MATRIX")
  }
  if (nrow(design) - ncol(design) < as.integer(minimum_residual_df)) {
    return("INSUFFICIENT_RESIDUAL_INFORMATION")
  }
  scaled <- design[, colnames(design) != "(Intercept)", drop = FALSE]
  if (ncol(scaled) > 1L) {
    scaled <- scale(scaled)
    condition_number <- tryCatch(kappa(scaled, exact = TRUE), error = function(error) Inf)
    if (!is.finite(condition_number) || condition_number > 1e8) {
      return("COLLINEAR_DESIGN_MATRIX")
    }
  }
  NULL
}

nrc_apply_factor_representations <- function(data, specification) {
  representations <- specification$factor_representations
  references <- list()
  if (is.null(representations) || !length(representations)) {
    return(list(data = data, factor_references = references))
  }
  for (factor_name in names(representations)) {
    representation <- representations[[factor_name]]
    if (is.null(data[[factor_name]])) {
      stop(sprintf("R_FACTOR_COLUMN_REQUIRED:%s", factor_name))
    }
    if (!(representation$data_type %in% c("categorical", "ordinal", "boolean"))) {
      next
    }
    reference_level <- representation$reference_level
    approved_levels <- as.character(unlist(representation$approved_levels))
    observed_levels <- unique(as.character(data[[factor_name]]))
    if (
      is.null(reference_level) || !nzchar(reference_level) ||
      !length(approved_levels) || !(reference_level %in% observed_levels)
    ) {
      stop(sprintf("R_REVIEWED_REFERENCE_LEVEL_UNAVAILABLE:%s", factor_name))
    }
    unknown_levels <- setdiff(observed_levels, approved_levels)
    if (length(unknown_levels)) {
      stop(sprintf("R_UNAPPROVED_FACTOR_LEVEL:%s", factor_name))
    }
    ordered_levels <- c(
      reference_level,
      setdiff(approved_levels, reference_level)
    )
    data[[factor_name]] <- factor(
      as.character(data[[factor_name]]),
      levels = ordered_levels,
      ordered = identical(representation$data_type, "ordinal")
    )
    references[[factor_name]] <- reference_level
  }
  list(data = data, factor_references = references)
}

nrc_apply_interval_metadata <- function(rows, specification) {
  interval_method <- specification$model_specification$interval_method
  if (is.null(interval_method)) {
    interval_method <- "wald_95"
  }
  lapply(rows, function(row) {
    lower <- row$conf.low
    upper <- row$conf.high
    if (is.null(lower)) {
      lower <- row$asymp.LCL
    }
    if (is.null(upper)) {
      upper <- row$asymp.UCL
    }
    if (is.null(lower)) {
      lower <- row$lower.CL
    }
    if (is.null(upper)) {
      upper <- row$upper.CL
    }
    if (
      !is.null(lower) && !is.null(upper) &&
      is.finite(as.numeric(lower)) && is.finite(as.numeric(upper))
    ) {
      row$conf.low <- as.numeric(lower)
      row$conf.high <- as.numeric(upper)
      row$interval_status <- "available"
      row$interval_method <- interval_method
      row$interval_unavailable_reason <- NULL
    } else {
      row$interval_status <- "unavailable"
      row$interval_method <- interval_method
      row$interval_unavailable_reason <- "INTERVAL_NOT_RETURNED_BY_MODEL_ENGINE"
    }
    row
  })
}

nrc_apply_multiplicity <- function(rows, specification) {
  multiplicity <- specification$multiplicity
  if (is.null(multiplicity)) {
    return(rows)
  }
  method <- multiplicity$method
  family_id <- multiplicity$family_id
  expected_test_ids <- as.character(unlist(multiplicity$expected_test_ids))
  decision_alpha <- multiplicity$alpha
  if (is.null(method) || !length(expected_test_ids) || is.null(decision_alpha)) {
    return(rows)
  }
  rows <- lapply(rows, function(row) {
    row$multiplicity_included <- FALSE
    row
  })
  test_ids <- vapply(rows, function(row) {
    if (!is.null(row$estimand_id) && nzchar(as.character(row$estimand_id))) {
      return(as.character(row$estimand_id))
    }
    if (!is.null(row$term) && nzchar(as.character(row$term))) {
      return(as.character(row$term))
    }
    ""
  }, character(1))
  indexes <- which(
    test_ids %in% expected_test_ids &
      vapply(
        rows,
        function(row) !is.null(row$p.value) && is.finite(as.numeric(row$p.value)),
        logical(1)
      )
  )
  if (!length(indexes)) {
    return(rows)
  }
  raw_values <- vapply(indexes, function(index) as.numeric(rows[[index]]$p.value), numeric(1))
  for (position in seq_along(indexes)) {
    index <- indexes[[position]]
    rows[[index]]$p.value_raw <- raw_values[[position]]
    rows[[index]]$p.value_adjusted <- NULL
    rows[[index]]$multiplicity_method <- method
    rows[[index]]$multiplicity_family_id <- family_id
    rows[[index]]$multiplicity_status <- "pending_central_reconciliation"
    rows[[index]]$multiplicity_included <- TRUE
    rows[[index]]$multiplicity_test_id <- test_ids[[index]]
    rows[[index]]$decision_alpha <- as.numeric(decision_alpha)
  }
  rows
}

nrc_multiplicity_reason <- function(rows, specification) {
  multiplicity <- specification$multiplicity
  if (is.null(multiplicity)) {
    return(NULL)
  }
  expected <- sort(unique(as.character(unlist(multiplicity$expected_test_ids))))
  observed <- sort(vapply(
    Filter(function(row) isTRUE(row$multiplicity_included), rows),
    function(row) as.character(row$multiplicity_test_id),
    character(1)
  ))
  if (!length(expected) || !identical(observed, expected)) {
    return("PREDECLARED_MULTIPLICITY_TEST_RESULT_MISMATCH")
  }
  NULL
}
