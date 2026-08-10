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
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime",
        multiplicity = list(
          method = "BH",
          family_id = "primary-yield-one-factor",
          family_scope_complete = TRUE
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
  p_rows <- Filter(function(row) !is.null(row$p.value), result$results)
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
})

test_that("variance-aware two-stage specifications consume positive first-stage weights", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "one_factor_inferential",
        engine = "r",
        support_gates_passed = TRUE,
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
