source(testthat::test_path("..", "..", "analysis", "stages", "contracts.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "diagnostics.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "mixed_models.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "marginal_contrasts.R"))

test_that("supported factors emit raw contrasts for central BH reconciliation", {
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "marginal_contrasts",
        candidate_id = "candidate-water",
        hypothesis_id = "H-water",
        engine = "r",
        support_gates_passed = TRUE,
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime",
        contrast_specification = list(
          factor_name = "water_regime",
          adjustment = "BH"
        ),
        multiplicity = list(method = "BH", family_id = "MF-primary")
      )
    ),
    data = data.frame(
      outcome = c(4.9, 5.1, 5.0, 6.1, 6.0, 6.2),
      water_regime = rep(c("rainfed", "irrigated"), each = 3L)
    )
  )
  result <- nrc_run_marginal_contrasts(stage)

  expect_identical(result$status, "completed")
  expect_true(length(result$results) >= 1L)
  expect_identical(result$metadata$multiplicity_method, "BH")
  expect_identical(result$metadata$multiple_testing_adjustment, "pending_central_reconciliation")
  expect_true(all(vapply(result$results, function(row) !is.null(row$p.value), logical(1))))
  expect_true(all(vapply(result$results, function(row) !is.null(row$p.value_raw), logical(1))))
  expect_true(all(vapply(result$results, function(row) is.null(row$p.value_adjusted), logical(1))))
  expect_true(all(vapply(
    result$results,
    function(row) identical(row$multiplicity_status, "pending_central_reconciliation"),
    logical(1)
  )))
})

test_that("missing contrast specifications remain explicit skips", {
  stage <- list(
    contract = list(
      specification = list(
        engine = "r",
        support_gates_passed = TRUE,
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ water_regime"
      )
    ),
    data = data.frame(
      outcome = c(5, 6),
      water_regime = c("rainfed", "irrigated")
    )
  )
  result <- nrc_run_marginal_contrasts(stage)

  expect_identical(result$status, "skipped")
  expect_identical(result$results[[1L]]$reason_codes[[1L]], "CONTRAST_SPECIFICATION_REQUIRED")
})
