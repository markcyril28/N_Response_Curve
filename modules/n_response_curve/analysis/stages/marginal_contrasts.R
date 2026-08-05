nrc_run_marginal_contrasts <- function(stage) {
  specification <- stage$contract$specification
  contrast_specification <- specification$contrast_specification
  if (is.null(contrast_specification) || !is.list(contrast_specification)) {
    return(nrc_skip_result("CONTRAST_SPECIFICATION_REQUIRED"))
  }
  if (!isTRUE(specification$support_gates_passed)) {
    return(nrc_skip_result("PRECOMPUTED_SUPPORT_GATE_REQUIRED"))
  }
  factor_name <- contrast_specification$factor_name
  adjustment <- contrast_specification$adjustment
  if (is.null(adjustment)) {
    adjustment <- "BH"
  }
  model_formula <- nrc_model_formula(specification, stage$data)
  if (is.null(model_formula)) {
    return(nrc_skip_result("MODEL_SPECIFICATION_REQUIRED"))
  }
  if (is.null(factor_name) || !is.character(factor_name) ||
        length(factor_name) != 1L || !factor_name %in% all.vars(model_formula)) {
    return(nrc_skip_result("CONTRAST_FACTOR_REQUIRED"))
  }

  model_kind <- specification$model_kind
  outcome_kind <- specification$outcome_kind
  first_stage_weights <- NULL
  first_stage <- specification$first_stage_uncertainty
  if (!is.null(first_stage)) {
    weight_column <- first_stage$weight_column
    if (is.null(weight_column) || !weight_column %in% names(stage$data)) {
      return(nrc_skip_result("FIRST_STAGE_WEIGHT_COLUMN_REQUIRED"))
    }
    first_stage_weights <- stage$data[[weight_column]]
    if (!is.numeric(first_stage_weights) ||
        any(!is.finite(first_stage_weights)) ||
        any(first_stage_weights <= 0)) {
      return(nrc_skip_result("FIRST_STAGE_WEIGHTS_INVALID"))
    }
  }
  fit <- nrc_fit_model(
    model_formula,
    stage$data,
    outcome_kind,
    model_kind,
    "Unsupported R model kind for marginal contrasts",
    weights = first_stage_weights
  )
  fitted <- fit$model
  warning_messages <- fit$warnings
  if (inherits(fitted, "error")) {
    return(nrc_failed_result(
      "R_CONTRAST_MODEL_FAILED",
      conditionMessage(fitted),
      list(warnings = as.list(unique(warning_messages)))
    ))
  }
  diagnostics <- nrc_model_diagnostics(fitted, warning_messages, nrow(stage$data))
  if (!isTRUE(diagnostics$converged)) {
    return(nrc_failed_result(
      "R_MODEL_NONCONVERGENCE",
      "R contrast model did not satisfy its convergence criteria",
      list(diagnostics = diagnostics)
    ))
  }
  if (isTRUE(diagnostics$singular) || isTRUE(diagnostics$boundary_fit)) {
    return(nrc_skip_result("R_MODEL_SINGULAR_OR_BOUNDARY", list(diagnostics = diagnostics)))
  }

  contrasted <- tryCatch(
    {
      if (requireNamespace("emmeans", quietly = TRUE)) {
        reference_grid <- emmeans::emmeans(
          fitted,
          specs = stats::as.formula(paste("~", factor_name))
        )
        raw <- broom::tidy(emmeans::contrast(reference_grid, method = "pairwise", adjust = "none"))
        raw$p.value_raw <- raw$p.value
        raw
      } else {
        if (!identical(outcome_kind, "continuous") || !identical(model_kind, "lm")) {
          nrc_abort("Fallback contrast estimation currently supports only continuous lm models without emmeans")
        }
        outcome_name <- all.vars(model_formula)[[1L]]
        analysis_frame <- stats::na.omit(stats::model.frame(fitted))
        factor_values <- analysis_frame[[factor_name]]
        outcome_values <- analysis_frame[[outcome_name]]
        if (!is.factor(factor_values)) {
          factor_values <- as.factor(factor_values)
        }
        if (!is.numeric(outcome_values)) {
          nrc_abort("Fallback contrast estimation requires a numeric outcome")
        }
        groups <- split(outcome_values, factor_values)
        if (length(groups) < 2L) {
          return(data.frame())
        }
        group_names <- names(groups)
        comparisons <- utils::combn(group_names, 2L, simplify = FALSE)
        if (!length(comparisons)) {
          return(data.frame())
        }
        raw_p <- vapply(
          comparisons,
          function(pair) {
            test <- tryCatch(
              stats::t.test(groups[[pair[[1L]]]], groups[[pair[[2L]]]]),
              error = function(error) error
            )
            if (inherits(test, "error") || !is.finite(as.numeric(test$p.value))) {
              NaN
            } else {
              as.numeric(test$p.value)
            }
          },
          numeric(1)
        )
        data.frame(
          contrast = vapply(
            comparisons,
            function(pair) paste(pair[[1L]], "vs", pair[[2L]]),
            character(1L)
          ),
          p.value = raw_p,
          p.value_raw = raw_p,
          stringsAsFactors = FALSE
        )
      }
    },
    error = function(error) error
  )
  if (inherits(contrasted, "error")) {
    return(nrc_failed_result("R_CONTRAST_ESTIMATION_FAILED", conditionMessage(contrasted)))
  }
  contrasted <- tryCatch(
    as.data.frame(contrasted),
    error = function(error) error
  )
  if (inherits(contrasted, "error")) {
    return(nrc_failed_result("R_CONTRAST_ESTIMATION_FAILED", conditionMessage(contrasted)))
  }
  if (nrow(contrasted) == 0L) {
    return(nrc_skip_result("CONTRAST_NOT_DEFINED_IN_DATA"))
  }
  if (!"p.value" %in% names(contrasted)) {
    return(nrc_failed_result("R_CONTRAST_ESTIMATION_FAILED", "Fallback contrast output missing p-values"))
  }

  results <- lapply(seq_len(nrow(contrasted)), function(index) {
    row <- as.list(contrasted[index, , drop = FALSE])
    row$factor_name <- factor_name
    row$adjustment <- adjustment
    if (is.null(row$p.value_raw)) {
      row$p.value_raw <- row$p.value
    }
    row$multiple_testing_adjustment <- "pending_central_reconciliation"
    row
  })
  results <- nrc_apply_multiplicity(results, specification)

  list(
    status = "completed",
    results = results,
    metadata = list(
      engine = "r",
      factor_name = factor_name,
      multiplicity_method = adjustment,
      multiple_testing_adjustment = "pending_central_reconciliation",
      model_kind = model_kind,
      outcome_kind = outcome_kind,
      diagnostics = diagnostics
    )
  )
}
