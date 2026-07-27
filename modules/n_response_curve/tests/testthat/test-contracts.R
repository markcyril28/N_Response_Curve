source(testthat::test_path("..", "..", "analysis", "stages", "contracts.R"))

test_that("R contract validates normalized parquet and stable keys", {
  skip_if_not_installed("arrow")
  stage_root <- tempfile("nrc-contract-")
  dir.create(stage_root)
  input_path <- file.path(stage_root, "input.parquet")
  output_path <- file.path(stage_root, "result.json")
  arrow::write_parquet(
    data.frame(response_series_uid = c("s1", "s2"), outcome = c(5, 6)),
    input_path
  )
  contract <- list(
    contract_version = 1L,
    contract_sha256 = "test-contract-hash",
    specification = list(analysis_family = "one_factor_inferential", engine = "r"),
    input = list(
      path = input_path,
      sha256 = nrc_sha256_file(input_path),
      row_count = 2L,
      stable_key = "response_series_uid",
      stable_key_count = 2L
    ),
    output = list(path = output_path)
  )
  contract_path <- file.path(stage_root, "contract.json")
  jsonlite::write_json(contract, contract_path, auto_unbox = TRUE)
  old_file_binding <- Sys.getenv("NRC_CONTRACT_FILE_SHA256", unset = NA_character_)
  old_semantic_binding <- Sys.getenv("NRC_CONTRACT_SHA256", unset = NA_character_)
  on.exit({
    if (is.na(old_file_binding)) Sys.unsetenv("NRC_CONTRACT_FILE_SHA256") else Sys.setenv(NRC_CONTRACT_FILE_SHA256 = old_file_binding)
    if (is.na(old_semantic_binding)) Sys.unsetenv("NRC_CONTRACT_SHA256") else Sys.setenv(NRC_CONTRACT_SHA256 = old_semantic_binding)
  }, add = TRUE)
  Sys.setenv(
    NRC_CONTRACT_FILE_SHA256 = nrc_sha256_file(contract_path),
    NRC_CONTRACT_SHA256 = contract$contract_sha256
  )
  stage <- nrc_read_contract(contract_path)

  expect_equal(nrow(stage$data), 2L)
  expect_equal(stage$contract$input$stable_key_count, 2L)
})

test_that("R contract rejects raw-source and operator configuration references", {
  expect_true(nrc_has_forbidden_contract_reference(list(operator_config_path = "scriptCONFIG.toml")))
  expect_false(nrc_has_forbidden_contract_reference(list(analysis_family = "one_factor_inferential")))
})
