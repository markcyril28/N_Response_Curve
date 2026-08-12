nrc_skip_result <- function(reason_code, metadata = list()) {
  list(
    status = "skipped",
    results = list(list(status = "skipped", reason_codes = list(reason_code))),
    metadata = c(list(engine = "r"), metadata)
  )
}

nrc_failed_result <- function(reason_code, message, metadata = list()) {
  list(
    status = "failed",
    results = list(list(
      status = "failed",
      reason_codes = list(reason_code),
      error_message = message
    )),
    metadata = c(list(engine = "r"), metadata)
  )
}

nrc_formula_has_hierarchy <- function(model_formula) {
  labels <- attr(stats::terms(model_formula), "term.labels")
  interactions <- labels[grepl(":", labels, fixed = TRUE)]
  if (length(interactions) == 0L) {
    return(TRUE)
  }
  all(vapply(interactions, function(interaction) {
    components <- strsplit(interaction, ":", fixed = TRUE)[[1L]]
    all(components %in% labels)
  }, logical(1)))
}

nrc_model_formula <- function(specification, data) {
  formula_text <- specification$model_formula
  if (is.null(formula_text) || !is.character(formula_text) || length(formula_text) != 1L || !nzchar(formula_text)) {
    return(NULL)
  }
  model_formula <- tryCatch(
    stats::as.formula(formula_text),
    error = function(error) NULL
  )
  if (is.null(model_formula) || !all(all.vars(model_formula) %in% names(data))) {
    return(NULL)
  }
  model_formula
}

nrc_fit_model <- function(
    model_formula,
    data,
    outcome_kind,
    model_kind,
    unsupported_message,
    require_binary_glm = FALSE,
    weights = NULL) {
  warning_messages <- character()
  fitted <- tryCatch(
    withCallingHandlers(
      {
        if (identical(outcome_kind, "continuous") && identical(model_kind, "lm")) {
          if (is.null(weights)) {
            stats::lm(model_formula, data = data)
          } else {
            weighted_data <- data
            weighted_data$.nrc_first_stage_weight <- weights
            stats::lm(
              model_formula,
              data = weighted_data,
              weights = .nrc_first_stage_weight
            )
          }
        } else if (identical(outcome_kind, "continuous") && identical(model_kind, "lmer")) {
          if (is.null(weights)) {
            lme4::lmer(model_formula, data = data, REML = FALSE)
          } else {
            weighted_data <- data
            weighted_data$.nrc_first_stage_weight <- weights
            lme4::lmer(
              model_formula,
              data = weighted_data,
              REML = FALSE,
              weights = .nrc_first_stage_weight
            )
          }
        } else if (identical(outcome_kind, "categorical") && identical(model_kind, "glm")) {
          outcome_name <- all.vars(model_formula)[[1L]]
          if (isTRUE(require_binary_glm) &&
                length(unique(stats::na.omit(data[[outcome_name]]))) != 2L) {
            nrc_abort("Categorical glm requires exactly two supported outcome levels")
          }
          stats::glm(model_formula, data = data, family = stats::binomial())
        } else if (identical(outcome_kind, "categorical") && identical(model_kind, "multinom")) {
          nnet::multinom(model_formula, data = data, trace = FALSE, Hess = TRUE, model = TRUE)
        } else if (identical(outcome_kind, "categorical") && identical(model_kind, "glmmTMB")) {
          glmmTMB::glmmTMB(model_formula, data = data, family = stats::binomial())
        } else {
          nrc_abort(unsupported_message)
        }
      },
      warning = function(warning) {
        warning_messages <<- c(warning_messages, conditionMessage(warning))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(error) error
  )
  list(model = fitted, warnings = warning_messages)
}

nrc_run_mixed_models <- function(stage) {
  specification <- stage$contract$specification
  if (!identical(specification$engine, "r")) {
    return(nrc_failed_result("ENGINE_ASSIGNMENT_MISMATCH", "R stage received a non-R specification"))
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
  if (!isTRUE(specification$support_gates_passed)) {
    return(nrc_skip_result("PRECOMPUTED_SUPPORT_GATE_REQUIRED"))
  }
  multiplicity <- specification$multiplicity
  if (!is.null(multiplicity) && !isTRUE(multiplicity$family_scope_complete)) {
    return(nrc_skip_result("MULTIPLICITY_FAMILY_RECONCILIATION_REQUIRED"))
  }
  if (!nrc_formula_has_hierarchy(model_formula)) {
    return(nrc_skip_result("INTERACTION_HIERARCHY_VIOLATION"))
  }
  minimum_residual_df <- specification$support_policy$minimum_residual_df
  if (is.null(minimum_residual_df) || !is.numeric(minimum_residual_df) ||
        length(minimum_residual_df) != 1L || !is.finite(minimum_residual_df) ||
        minimum_residual_df < 1 || minimum_residual_df != as.integer(minimum_residual_df)) {
    return(nrc_skip_result("PREDECLARED_RESIDUAL_DF_THRESHOLD_REQUIRED"))
  }
  design_reason <- nrc_fixed_effect_design_reason(
    model_formula,
    analysis_data,
    as.integer(minimum_residual_df)
  )
  if (!is.null(design_reason)) {
    return(nrc_skip_result(design_reason))
  }
  model_kind <- specification$model_kind
  if (is.null(model_kind)) {
    model_kind <- "lm"
  }
  outcome_kind <- specification$outcome_kind
  if (is.null(model_kind) || is.null(outcome_kind)) {
    return(nrc_skip_result("PREDECLARED_MODEL_SPECIFICATION_REQUIRED"))
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
    stage$data,
    outcome_kind,
    model_kind,
    "Unsupported R model kind for the declared outcome type",
    require_binary_glm = TRUE,
    weights = first_stage_weights
  )
  fitted <- fit$model
  warning_messages <- fit$warnings
  if (inherits(fitted, "error")) {
    return(nrc_failed_result(
      "R_MODEL_FIT_FAILED",
      conditionMessage(fitted),
      list(warning_messages = as.list(unique(warning_messages)))
    ))
  }
  diagnostics <- nrc_model_diagnostics(fitted, warning_messages, nrow(stage$data))
  if (!isTRUE(diagnostics$converged)) {
    return(nrc_failed_result(
      "R_MODEL_NONCONVERGENCE",
      "R model did not satisfy its convergence criteria",
      list(diagnostics = diagnostics)
    ))
  }
  if (isTRUE(diagnostics$singular) || isTRUE(diagnostics$boundary_fit)) {
    return(nrc_skip_result("R_MODEL_SINGULAR_OR_BOUNDARY", list(diagnostics = diagnostics)))
  }
  results <- nrc_apply_multiplicity(nrc_tidy_model(fitted), specification)
  if (!length(results)) {
    return(nrc_failed_result("R_MODEL_TIDY_RESULT_EMPTY", "R model produced no reportable fixed-effect rows"))
  }
  estimates_are_finite <- vapply(results, function(row) {
    is.null(row$estimate) || is.finite(as.numeric(row$estimate))
  }, logical(1))
  if (!all(estimates_are_finite)) {
    return(nrc_failed_result("R_MODEL_NONFINITE_ESTIMATE", "R model produced a nonfinite fixed-effect estimate"))
  }
  list(
    status = "completed",
    results = results,
    metadata = list(
      engine = "r",
      model_kind = model_kind,
      outcome_kind = outcome_kind,
      multiplicity = specification$multiplicity,
      diagnostics = diagnostics
    )
  )
}
