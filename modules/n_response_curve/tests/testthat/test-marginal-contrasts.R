source(testthat::test_path("..", "..", "analysis", "stages", "contracts.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "diagnostics.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "mixed_models.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "marginal_contrasts.R"))

test_that("supported factors emit raw contrasts for central BH reconciliation", {
  skip_if_not_installed("emmeans")
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
          estimand_id = "estimand-water-v1",
          factor_name = "water_regime",
          treatment = "irrigated",
          comparator = "rainfed",
          direction = "irrigated_minus_rainfed",
          adjustment = "BH"
        ),
        multiplicity = list(
          method = "BH",
          family_id = "MF-primary",
          alpha = 0.05,
          expected_test_ids = list("estimand-water-v1")
        )
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
  expect_true(all(vapply(
    result$results,
    function(row) isTRUE(row$multiplicity_included),
    logical(1)
  )))
  expect_true(all(vapply(
    result$results,
    function(row) identical(row$multiplicity_test_id, "estimand-water-v1"),
    logical(1)
  )))
  expect_true(all(vapply(
    result$results,
    function(row) identical(row$decision_alpha, 0.05),
    logical(1)
  )))
})

test_that("prespecified management estimands emit only the requested direction", {
  skip_if_not_installed("emmeans")
  stage <- list(
    contract = list(
      specification = list(
        analysis_family = "marginal_contrasts",
        candidate_id = "candidate-management",
        hypothesis_id = "H-management",
        engine = "r",
        support_gates_passed = TRUE,
        model_kind = "lm",
        outcome_kind = "continuous",
        model_formula = "outcome ~ recommendation_class",
        contrast_specification = list(
          estimand_id = "estimand-management-v1",
          factor_name = "recommendation_class",
          treatment = "RCM",
          comparator = "NOPT_NPK",
          direction = "RCM_minus_NOPT_NPK",
          target_population = "verified_resolved_philippine_trials",
          same_context_required = TRUE,
          dependence_unit = "comparison_set_uid",
          adjustment = "BH"
        ),
        multiplicity = list(
          method = "BH",
          family_id = "MF-management",
          alpha = 0.05,
          expected_test_ids = list("estimand-management-v1")
        )
      )
    ),
    data = data.frame(
      outcome = c(5.0, 6.0, 5.2, 6.2, 5.4, 6.4, 5.6, 6.6),
      recommendation_class = rep(c("NOPT_NPK", "RCM"), 4L),
      comparison_set_uid = rep(paste0("comparison-", seq_len(4L)), each = 2L)
    )
  )

  result <- nrc_run_marginal_contrasts(stage)

  expect_identical(result$status, "completed")
  expect_length(result$results, 1L)
  expect_identical(result$results[[1L]]$estimand_id, "estimand-management-v1")
  expect_identical(result$results[[1L]]$treatment, "RCM")
  expect_identical(result$results[[1L]]$comparator, "NOPT_NPK")
  expect_identical(result$results[[1L]]$direction, "RCM_minus_NOPT_NPK")
  expect_gt(result$results[[1L]]$estimate, 0)
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

test_that("an unavailable model-based contrast engine never falls back to an unpaired t-test", {
  original <- nrc_emmeans_available
  assign(
    "nrc_emmeans_available",
    function() FALSE,
    envir = environment(nrc_run_marginal_contrasts)
  )
  on.exit(
    assign(
      "nrc_emmeans_available",
      original,
      envir = environment(nrc_run_marginal_contrasts)
    ),
    add = TRUE
  )
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
          estimand_id = "estimand-water-v1",
          factor_name = "water_regime",
          treatment = "irrigated",
          comparator = "rainfed",
          direction = "irrigated_minus_rainfed",
          adjustment = "BH"
        )
      )
    ),
    data = data.frame(
      outcome = c(4.9, 5.1, 5.0, 6.1, 6.0, 6.2),
      water_regime = rep(c("rainfed", "irrigated"), each = 3L)
    )
  )

  result <- nrc_run_marginal_contrasts(stage)

  expect_identical(result$status, "skipped")
  expect_identical(
    result$results[[1L]]$reason_codes[[1L]],
    "MODEL_BASED_CONTRAST_ENGINE_UNAVAILABLE"
  )
})
