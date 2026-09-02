source(testthat::test_path("..", "..", "analysis", "stages", "contracts.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "diagnostics.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "mixed_models.R"))

test_that("supported continuous R formulas produce tidy model output", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "one_factor_inferential",
        engine = "r",
        support_gates_passed = TRUE,
        support_policy = list(minimum_residual_df = 3L),
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime",
        model_specification = list(
          interval_method = "wald_95",
          multiplicity_test_ids = list("water_regimerainfed")
        ),
        factor_representations = list(
          water_regime = list(
            data_type = "categorical",
            reference_level = "irrigated",
            approved_levels = list("irrigated", "rainfed")
          )
        ),
        multiplicity = list(
          method = "BH",
          family_id = "primary-yield-one-factor",
          family_scope_complete = TRUE,
          alpha = 0.05,
          expected_test_ids = list("water_regimerainfed")
        )
      )
    ),
    data = data.frame(
      outcome = c(5, 6, 7, 8, 9, 10),
      water_regime = rep(c("rainfed", "irrigated"), each = 3L)
    )
  )
  result <- nrc_run_mixed_models(stage)

  expect_identical(result$status, "completed")
  expect_true(length(result$results) >= 2L)
  expect_identical(result$metadata$engine, "r")
  expect_true(all(c(
    "converged", "singular", "boundary_fit", "dropped_row_count",
    "residual_summary", "influence", "contrast_coding"
  ) %in% names(result$metadata$diagnostics)))
  expect_identical(result$metadata$diagnostics$dropped_row_count, 0L)
  p_rows <- Filter(
    function(row) isTRUE(row$multiplicity_included),
    result$results
  )
  expect_true(all(vapply(p_rows, function(row) !is.null(row$p.value_raw), logical(1))))
  expect_true(all(vapply(p_rows, function(row) is.null(row$p.value_adjusted), logical(1))))
  expect_true(all(vapply(
    p_rows,
    function(row) identical(row$multiplicity_status, "pending_central_reconciliation"),
    logical(1)
  )))
  expect_true(all(vapply(
    p_rows,
    function(row) identical(row$multiplicity_family_id, "primary-yield-one-factor"),
    logical(1)
  )))
  by_term <- stats::setNames(result$results, vapply(
    result$results,
    function(row) row$term,
    character(1)
  ))
  expect_false(by_term[["(Intercept)"]]$multiplicity_included)
  expect_true(by_term[["water_regimerainfed"]]$multiplicity_included)
  expect_identical(
    by_term[["water_regimerainfed"]]$multiplicity_test_id,
    "water_regimerainfed"
  )
  expect_identical(by_term[["water_regimerainfed"]]$decision_alpha, 0.05)
  expect_identical(by_term[["water_regimerainfed"]]$interval_status, "available")
  expect_true(all(c("conf.low", "conf.high") %in% names(by_term[["water_regimerainfed"]])))
})

test_that("reviewed categorical reference levels are applied before model fitting", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "one_factor_inferential",
        engine = "r",
        support_gates_passed = TRUE,
        support_policy = list(minimum_residual_df = 3L),
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime",
        factor_representations = list(
          water_regime = list(
            data_type = "categorical",
            reference_level = "irrigated",
            approved_levels = list("irrigated", "rainfed")
          )
        )
      )
    ),
    data = data.frame(
      outcome = c(4.9, 5.1, 5.0, 6.1, 6.0, 6.2),
      water_regime = rep(c("rainfed", "irrigated"), each = 3L)
    )
  )

  result <- nrc_run_mixed_models(stage)

  expect_identical(result$status, "completed")
  expect_true("water_regimerainfed" %in% vapply(
    result$results,
    function(row) row$term,
    character(1)
  ))
  expect_false("water_regimeirrigated" %in% vapply(
    result$results,
    function(row) row$term,
    character(1)
  ))
  expect_identical(
    result$metadata$factor_references$water_regime,
    "irrigated"
  )
})

test_that("variance-aware two-stage specifications consume positive first-stage weights", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "one_factor_inferential",
        engine = "r",
        support_gates_passed = TRUE,
        support_policy = list(minimum_residual_df = 3L),
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime",
        first_stage_uncertainty = list(weight_column = "first_stage_weight")
      )
    ),
    data = data.frame(
      outcome = c(5, 6, 7, 8, 9, 10, 11, 12),
      water_regime = rep(c("rainfed", "irrigated"), each = 4L),
      first_stage_weight = c(4, 4, 1, 1, 1, 1, 4, 4)
    )
  )

  result <- nrc_run_mixed_models(stage)

  expect_identical(result$status, "completed")
  expect_true(length(result$results) >= 2L)

  stage$data$first_stage_weight[[1L]] <- 0
  invalid <- nrc_run_mixed_models(stage)
  expect_identical(invalid$status, "skipped")
  expect_identical(
    invalid$results[[1L]]$reason_codes[[1L]],
    "FIRST_STAGE_WEIGHTS_INVALID"
  )
})

test_that("supported multiclass outcomes use an explicit multinomial model", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "one_factor_inferential",
        engine = "r",
        support_gates_passed = TRUE,
        support_policy = list(minimum_residual_df = 3L),
        model_kind = "multinom",
        outcome_kind = "categorical",
        model_formula = "outcome ~ water_regime"
      )
    ),
    data = data.frame(
      outcome = rep(c("linear", "quadratic", "plateau"), each = 4L),
      water_regime = rep(c("rainfed", "irrigated"), 6L)
    )
  )
  result <- nrc_run_mixed_models(stage)

  expect_identical(result$status, "completed")
  expect_true(length(result$results) >= 2L)
  expect_identical(result$metadata$model_kind, "multinom")
})

test_that("missing formulas remain explicit skips", {
  stage <- list(
    contract = list(specification = list(engine = "r")),
    data = data.frame(outcome = c(5, 6))
  )
  result <- nrc_run_mixed_models(stage)

  expect_identical(result$status, "skipped")
  expect_identical(result$results[[1L]]$reason_codes[[1L]], "MODEL_SPECIFICATION_REQUIRED")
})

test_that("the reviewed residual-df threshold controls R model dispatch", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "one_factor_inferential",
        engine = "r",
        support_gates_passed = TRUE,
        support_policy = list(minimum_residual_df = 5L),
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime"
      )
    ),
    data = data.frame(
      outcome = c(5, 6, 7, 8, 9, 10),
      water_regime = rep(c("rainfed", "irrigated"), each = 3L)
    )
  )

  result <- nrc_run_mixed_models(stage)

  expect_identical(result$status, "skipped")
  expect_identical(
    result$results[[1L]]$reason_codes[[1L]],
    "INSUFFICIENT_RESIDUAL_INFORMATION"
  )
})

test_that("the convergence gate reads a glm's own convergence flag", {
  # A glm carries class c("glm", "lm"), so an `inherits(model, "lm")` fallback
  # answers for it too and reports an unconverged fit as converged.
  separable <- data.frame(
    outcome = c(0, 0, 0, 0, 1, 1, 1, 1),
    predictor = c(1, 2, 3, 4, 10, 11, 12, 13)
  )
  stopped_early <- suppressWarnings(stats::glm(
    outcome ~ predictor,
    data = separable,
    family = stats::binomial(),
    control = list(maxit = 1L)
  ))
  expect_false(isTRUE(stopped_early$converged))
  expect_false(nrc_model_converged(stopped_early))

  converged_fit <- stats::glm(
    outcome ~ predictor,
    data = data.frame(
      outcome = c(0, 1, 0, 1, 0, 1, 0, 1),
      predictor = c(1, 2, 3, 4, 5, 6, 7, 8)
    ),
    family = stats::binomial()
  )
  expect_true(nrc_model_converged(converged_fit))

  # An lm has no iteration to converge, so it stays reported as converged.
  expect_true(nrc_model_converged(stats::lm(outcome ~ predictor, data = separable)))
})
