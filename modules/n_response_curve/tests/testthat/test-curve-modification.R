source(testthat::test_path("..", "..", "analysis", "stages", "contracts.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "diagnostics.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "mixed_models.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "curve_modification.R"))

test_that("curve-modification specifications require an N-rate term", {
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
    data = data.frame(outcome = c(5, 6), water_regime = c("rainfed", "irrigated"))
  )
  result <- nrc_run_curve_modification(stage)

  expect_identical(result$status, "skipped")
  expect_identical(result$results[[1L]]$reason_codes[[1L]], "N_RATE_TERM_REQUIRED")
})
