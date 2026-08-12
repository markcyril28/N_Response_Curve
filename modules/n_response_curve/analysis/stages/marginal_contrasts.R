nrc_emmeans_available <- function() {
  requireNamespace("emmeans", quietly = TRUE)
}

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
  treatment <- contrast_specification$treatment
  comparator <- contrast_specification$comparator
  exact_estimand <- !is.null(treatment) || !is.null(comparator)
  adjustment <- contrast_specification$adjustment
  if (is.null(adjustment)) {
    adjustment <- "BH"
  }
  represented <- tryCatch(
    nrc_apply_factor_representations(stage$data, specification),
    error = function(error) error
  )
  if (inherits(represented, "error")) {
    return(nrc_skip_result(conditionMessage(represented)))
  }
  analysis_data <- represented$data
  model_formula <- nrc_model_formula(specification, analysis_data)
  if (is.null(model_formula)) {
    return(nrc_skip_result("MODEL_SPECIFICATION_REQUIRED"))
  }
  if (is.null(factor_name) || !is.character(factor_name) ||
        length(factor_name) != 1L || !factor_name %in% all.vars(model_formula)) {
    return(nrc_skip_result("CONTRAST_FACTOR_REQUIRED"))
  }
  if (isTRUE(exact_estimand)) {
    if (is.null(treatment) || is.null(comparator) ||
          !is.character(treatment) || !is.character(comparator) ||
          length(treatment) != 1L || length(comparator) != 1L ||
          !nzchar(treatment) || !nzchar(comparator) || identical(treatment, comparator)) {
      return(nrc_skip_result("PRESPECIFIED_MANAGEMENT_CONTRAST_REQUIRED"))
    }
    observed_levels <- unique(as.character(analysis_data[[factor_name]]))
    if (!all(c(treatment, comparator) %in% observed_levels)) {
      return(nrc_skip_result("PRESPECIFIED_CONTRAST_LEVELS_UNAVAILABLE"))
    }
    if (isTRUE(contrast_specification$same_context_required)) {
      dependence_unit <- contrast_specification$dependence_unit
      if (is.null(dependence_unit) || !is.character(dependence_unit) ||
            length(dependence_unit) != 1L || !dependence_unit %in% names(analysis_data)) {
        return(nrc_skip_result("VERIFIED_SAME_CONTEXT_ESTIMAND_REQUIRED"))
      }
      context_levels <- split(
        as.character(analysis_data[[factor_name]]),
        as.character(analysis_data[[dependence_unit]])
      )
      if (!length(context_levels) || any(vapply(
        context_levels,
        function(levels) !all(c(treatment, comparator) %in% levels),
        logical(1)
      ))) {
        return(nrc_skip_result("PRESPECIFIED_CONTRAST_PAIR_INCOMPLETE"))
      }
    }
  }

  model_kind <- specification$model_kind
  outcome_kind <- specification$outcome_kind
  if (is.null(model_kind) || is.null(outcome_kind)) {
    return(nrc_skip_result("PREDECLARED_MODEL_SPECIFICATION_REQUIRED"))
  }
  if (!nrc_emmeans_available()) {
    return(nrc_skip_result("MODEL_BASED_CONTRAST_ENGINE_UNAVAILABLE"))
  }
  first_stage_weights <- NULL
  first_stage <- specification$first_stage_uncertainty
  if (!is.null(first_stage)) {
    weight_column <- first_stage$weight_column
    if (is.null(weight_column) || !weight_column %in% names(analysis_data)) {
      return(nrc_skip_result("FIRST_STAGE_WEIGHT_COLUMN_REQUIRED"))
    }
    first_stage_weights <- analysis_data[[weight_column]]
    if (!is.numeric(first_stage_weights) ||
        any(!is.finite(first_stage_weights)) ||
        any(first_stage_weights <= 0)) {
      return(nrc_skip_result("FIRST_STAGE_WEIGHTS_INVALID"))
    }
  }
  fit <- nrc_fit_model(
    model_formula,
    analysis_data,
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
  diagnostics <- nrc_model_diagnostics(fitted, warning_messages, nrow(analysis_data))
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
      reference_grid <- emmeans::emmeans(
        fitted,
        specs = stats::as.formula(paste("~", factor_name))
      )
      if (isTRUE(exact_estimand)) {
        grid_levels <- as.character(as.data.frame(reference_grid)[[factor_name]])
        contrast_weights <- rep(0, length(grid_levels))
        contrast_weights[grid_levels == treatment] <- 1
        contrast_weights[grid_levels == comparator] <- -1
        method <- list()
        contrast_label <- contrast_specification$direction
        if (is.null(contrast_label) || !is.character(contrast_label) ||
              length(contrast_label) != 1L || !nzchar(contrast_label)) {
          contrast_label <- paste(treatment, "minus", comparator)
        }
        method[[contrast_label]] <- contrast_weights
        contrast_result <- emmeans::contrast(
          reference_grid,
          method = method,
          adjust = "none"
        )
      } else {
        contrast_result <- emmeans::contrast(
          reference_grid,
          method = "pairwise",
          adjust = "none"
        )
      }
      raw <- as.data.frame(summary(
        contrast_result,
        infer = c(TRUE, TRUE),
        adjust = "none"
      ))
      raw$p.value_raw <- raw$p.value
      raw
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
    if (isTRUE(exact_estimand)) {
      row$estimand_id <- contrast_specification$estimand_id
      row$treatment <- treatment
      row$comparator <- comparator
      row$direction <- contrast_specification$direction
      row$target_population <- contrast_specification$target_population
      row$dependence_unit <- contrast_specification$dependence_unit
    }
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
      estimand_id = contrast_specification$estimand_id,
      exact_estimand = exact_estimand,
      multiplicity_method = adjustment,
      multiple_testing_adjustment = "pending_central_reconciliation",
      model_kind = model_kind,
      outcome_kind = outcome_kind,
      diagnostics = diagnostics
    )
  )
}
