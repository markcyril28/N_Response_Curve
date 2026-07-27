testthat::test_that("exact fixed-effect gates reject aliased interactions", {
  source(testthat::test_path("..", "..", "analysis", "stages", "diagnostics.R"))
  data <- data.frame(
    outcome = seq_len(12),
    first = rep(c("a", "b"), 6),
    second = rep(c("x", "y"), 6)
  )
  data$second <- ifelse(data$first == "a", "x", "y")
  reason <- nrc_fixed_effect_design_reason(outcome ~ first * second, data, 3L)
  testthat::expect_identical(reason, "ALIASED_DESIGN_MATRIX")
})

testthat::test_that("completed model results cannot be empty", {
  source(testthat::test_path("..", "..", "analysis", "stages", "mixed_models.R"))
  result <- nrc_failed_result("TEST_REASON", "test failure")
  testthat::expect_identical(result$status, "failed")
  testthat::expect_length(result$results, 1L)
})
