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

nrc_run_mixed_models <- function(stage) {
  specification <- stage$contract$specification
  if (!identical(specification$engine, "r")) {
    return(nrc_failed_result("ENGINE_ASSIGNMENT_MISMATCH", "R stage received a non-R specification"))
  }
  model_formula <- nrc_model_formula(specification, stage$data)
  if (is.null(model_formula)) {
    return(nrc_skip_result("MODEL_SPECIFICATION_REQUIRED"))
  }
  if (!isTRUE(specification$support_gates_passed)) {
    return(nrc_skip_result("PRECOMPUTED_SUPPORT_GATE_REQUIRED"))
  }
  if (!nrc_formula_has_hierarchy(model_formula)) {
    return(nrc_skip_result("INTERACTION_HIERARCHY_VIOLATION"))
  }
  model_kind <- specification$model_kind
  if (is.null(model_kind)) {
    model_kind <- "lm"
  }
  outcome_kind <- specification$outcome_kind
  if (is.null(outcome_kind)) {
    outcome_kind <- "continuous"
  }
  warning_messages <- character()
  fitted <- tryCatch(
    withCallingHandlers(
      {
        if (identical(outcome_kind, "continuous") && identical(model_kind, "lm")) {
          stats::lm(model_formula, data = stage$data)
        } else if (identical(outcome_kind, "continuous") && identical(model_kind, "lmer")) {
          lme4::lmer(model_formula, data = stage$data, REML = FALSE)
        } else if (identical(outcome_kind, "categorical") && identical(model_kind, "glm")) {
          outcome_name <- all.vars(model_formula)[[1L]]
          if (length(unique(stats::na.omit(stage$data[[outcome_name]]))) != 2L) {
            nrc_abort("Categorical glm requires exactly two supported outcome levels")
          }
          stats::glm(model_formula, data = stage$data, family = stats::binomial())
        } else if (identical(outcome_kind, "categorical") && identical(model_kind, "multinom")) {
          nnet::multinom(model_formula, data = stage$data, trace = FALSE, Hess = TRUE, model = TRUE)
        } else if (identical(outcome_kind, "categorical") && identical(model_kind, "glmmTMB")) {
          glmmTMB::glmmTMB(model_formula, data = stage$data, family = stats::binomial())
        } else {
          nrc_abort("Unsupported R model kind for the declared outcome type")
        }
      },
      warning = function(warning) {
        warning_messages <<- c(warning_messages, conditionMessage(warning))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(error) error
  )
  if (inherits(fitted, "error")) {
    return(nrc_failed_result(
      "R_MODEL_FIT_FAILED",
      conditionMessage(fitted),
      list(warning_messages = as.list(unique(warning_messages)))
    ))
  }
  results <- nrc_apply_multiplicity(nrc_tidy_model(fitted), specification)
  list(
    status = "completed",
    results = results,
    metadata = list(
      engine = "r",
      model_kind = model_kind,
      outcome_kind = outcome_kind,
      multiplicity = specification$multiplicity,
      diagnostics = nrc_model_diagnostics(fitted, warning_messages, nrow(stage$data))
    )
  )
}
