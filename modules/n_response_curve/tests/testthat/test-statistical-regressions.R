source(testthat::test_path("..", "..", "analysis", "stages", "contracts.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "diagnostics.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "mixed_models.R"))
source(testthat::test_path("..", "..", "analysis", "stages", "marginal_contrasts.R"))

test_that("lmer produces the declared Wald tests for multiplicity reconciliation", {
  skip_if_not_installed("lme4")
  skip_if_not_installed("broom.mixed")
  set.seed(24)
  data <- data.frame(
    study = factor(rep(seq_len(10), each = 8)),
    x = rep(seq(-1, 1, length.out = 8), 10)
  )
  data$y <- 2 + 0.6 * data$x + rep(rnorm(10), each = 8) + rnorm(80, sd = 0.5)
  specification <- list(
    engine = "r", support_gates_passed = TRUE,
    support_policy = list(minimum_residual_df = 3L),
    model_kind = "lmer", outcome_kind = "continuous",
    model_formula = "y ~ x + (1 | study)",
    model_specification = list(interval_method = "wald_95"),
    multiplicity = list(
      method = "BH", family_id = "regression", family_scope_complete = TRUE,
      alpha = 0.05, expected_test_ids = list("x")
    )
  )
  result <- nrc_run_mixed_models(list(contract = list(specification = specification), data = data))
  expect_identical(result$status, "completed")
  slope <- Filter(function(row) identical(row$term, "x"), result$results)[[1L]]
  expect_true(slope$multiplicity_included)
  expect_identical(slope$inference_distribution, "normal")
  expect_equal(slope$p.value_raw, 2 * pnorm(-abs(slope$estimate / slope$std.error)))
  expect_equal(slope$conf.high - slope$estimate, qnorm(0.975) * slope$std.error)
})

test_that("GLM intervals match Wald p-values instead of profile intervals", {
  skip_if_not_installed("broom")
  data <- data.frame(x = rep(seq(-1, 1, length.out = 10), 4), y = rep(c(0, 1, 0, 0, 1), 8))
  fit <- glm(y ~ x, data = data, family = binomial())
  rows <- nrc_tidy_model(fit)
  slope <- rows[[2L]]
  expect_equal(slope$conf.high - slope$estimate, qnorm(0.975) * slope$std.error)
  expect_equal(slope$p.value, 2 * pnorm(-abs(slope$estimate / slope$std.error)))
})

test_that("ordinal predictors retain the declared reference contrasts", {
  represented <- nrc_apply_factor_representations(
    data.frame(f = rep(c("low", "medium", "high"), each = 3)),
    list(factor_representations = list(f = list(
      data_type = "ordinal", reference_level = "medium",
      approved_levels = c("low", "medium", "high")
    )))
  )
  design <- model.matrix(~ f, represented$data)
  expect_setequal(colnames(design), c("(Intercept)", "flow", "fhigh"))
  expect_true(all(design[represented$data$f == "medium", -1L] == 0))
})

test_that("three-way interactions require every lower-order interaction", {
  expect_false(nrc_formula_has_hierarchy(y ~ a + b + c + a:b:c))
  expect_false(nrc_formula_has_hierarchy(y ~ a + b + c + a:b + a:c + a:b:c))
  expect_true(nrc_formula_has_hierarchy(y ~ a * b * c))
})

test_that("complete separation is a boundary fit even after IRLS converges", {
  fit <- suppressWarnings(glm(
    y ~ x, data = data.frame(y = rep(0:1, each = 10), x = 1:20),
    family = binomial(), control = glm.control(maxit = 100)
  ))
  expect_true(fit$converged)
  expect_true(nrc_model_diagnostics(fit)$boundary_fit)
})

test_that("binomial fits accept categorical labels and reject a third class", {
  data <- data.frame(x = 1:12, y = rep(c("absent", "present"), 6))
  fitted <- nrc_fit_model(y ~ x, data, "categorical", "glm", "unsupported")$model
  expect_s3_class(fitted, "glm")
  reference <- glm(I(y == "present") ~ x, data = data, family = binomial())
  expect_equal(unname(coef(fitted)), unname(coef(reference)))
  data$y[[1L]] <- "third"
  invalid <- nrc_fit_model(y ~ x, data, "categorical", "glm", "unsupported")$model
  expect_s3_class(invalid, "error")
})

test_that("marginal contrasts enforce the same residual-information gate as models", {
  skip_if_not_installed("emmeans")
  stage <- list(
    data = data.frame(y = c(1, 2, 3, 5, 6, 7), f = rep(c("a", "b"), each = 3)),
    contract = list(specification = list(
      engine = "r", support_gates_passed = TRUE,
      support_policy = list(minimum_residual_df = 5L),
      model_kind = "lm", outcome_kind = "continuous", model_formula = "y ~ f",
      contrast_specification = list(factor_name = "f", treatment = "b", comparator = "a")
    ))
  )
  result <- nrc_run_marginal_contrasts(stage)
  expect_identical(result$status, "skipped")
  expect_identical(result$results[[1L]]$reason_codes[[1L]], "INSUFFICIENT_RESIDUAL_INFORMATION")
})

test_that("R result serialization preserves tiny and threshold-adjacent p-values", {
  skip_if_not_installed("jsonlite")
  output <- tempfile(fileext = ".json")
  on.exit(unlink(output))
  p_values <- c(3.456789123456e-9, 0.04999987654321)
  stage <- list(
    contract = list(contract_sha256 = "contract", input = list(sha256 = "input", row_count = 2)),
    output_path = output
  )
  result <- list(status = "completed", metadata = list(), results = lapply(p_values, function(p) list(p.value = p)))
  nrc_write_stage_result(stage, result)
  restored <- jsonlite::read_json(output)
  actual <- vapply(restored$results, function(row) row$p.value, numeric(1))
  expect_equal(actual, p_values, tolerance = 1e-15)
  expect_gt(actual[[1L]], 0)
  expect_lt(actual[[2L]], 0.05)
})
