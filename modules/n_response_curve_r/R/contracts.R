nrc_abort <- function(message) {
  stop(message, call. = FALSE)
}

nrc_sha256_file <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    nrc_abort("R contract requires the digest package for SHA-256 verification")
  }
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

nrc_is_within_directory <- function(path, root) {
  normalized_path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  normalized_root <- normalizePath(root, winslash = "/", mustWork = TRUE)
  identical(normalized_path, normalized_root) || startsWith(
    normalized_path,
    paste0(normalized_root, "/")
  )
}

nrc_has_forbidden_contract_reference <- function(value) {
  forbidden_fragments <- c(
    "operator_config",
    "scriptconfig",
    "raw_source",
    "source_file_path",
    "toml_path"
  )
  if (is.list(value)) {
    names_value <- names(value)
    if (!is.null(names_value)) {
      normalized_names <- gsub("-", "_", tolower(names_value), fixed = TRUE)
      if (any(vapply(
        forbidden_fragments,
        function(fragment) any(grepl(fragment, normalized_names, fixed = TRUE)),
        logical(1)
      ))) {
        return(TRUE)
      }
    }
    return(any(vapply(value, nrc_has_forbidden_contract_reference, logical(1))))
  }
  FALSE
}

nrc_parse_contract_path <- function(arguments) {
  contract_index <- match("--contract", arguments)
  if (is.na(contract_index) || contract_index == length(arguments)) {
    nrc_abort("R entry point requires exactly one --contract <path> argument")
  }
  if (length(arguments) != 2L || contract_index != 1L) {
    nrc_abort("R entry point accepts no arguments other than --contract <path>")
  }
  arguments[[contract_index + 1L]]
}

nrc_frame_to_rows <- function(frame) {
  if (!is.data.frame(frame) || nrow(frame) == 0L) {
    return(list())
  }
  lapply(seq_len(nrow(frame)), function(index) as.list(frame[index, , drop = FALSE]))
}

nrc_read_contract <- function(contract_path) {
  normalized_contract_path <- normalizePath(contract_path, winslash = "/", mustWork = TRUE)
  stage_root <- dirname(normalized_contract_path)
  contract <- jsonlite::read_json(normalized_contract_path, simplifyVector = FALSE)
  required_names <- c("contract_version", "contract_sha256", "specification", "input", "output")
  if (!is.list(contract) || !all(required_names %in% names(contract))) {
    nrc_abort("R stage contract has missing required fields")
  }
  if (!identical(as.integer(contract$contract_version), 1L)) {
    nrc_abort("R stage contract has an unsupported version")
  }
  if (nrc_has_forbidden_contract_reference(contract$specification)) {
    nrc_abort("R stage contract contains a forbidden raw-source or operator-config reference")
  }
  input <- contract$input
  output <- contract$output
  required_input <- c("path", "sha256", "row_count", "stable_key", "stable_key_count")
  if (!is.list(input) || !all(required_input %in% names(input))) {
    nrc_abort("R stage contract input descriptor is incomplete")
  }
  if (!is.list(output) || !"path" %in% names(output)) {
    nrc_abort("R stage contract output descriptor is incomplete")
  }
  input_path <- normalizePath(input$path, winslash = "/", mustWork = TRUE)
  output_path <- normalizePath(output$path, winslash = "/", mustWork = FALSE)
  if (!nrc_is_within_directory(input_path, stage_root) ||
      !nrc_is_within_directory(output_path, stage_root)) {
    nrc_abort("R contract input and output must remain inside the isolated stage root")
  }
  if (!identical(nrc_sha256_file(input_path), input$sha256)) {
    nrc_abort("R stage input SHA-256 does not match the contract")
  }
  data <- arrow::read_parquet(input_path, as_data_frame = TRUE)
  if (!identical(nrow(data), as.integer(input$row_count))) {
    nrc_abort("R stage input row count does not match the contract")
  }
  stable_key <- as.character(input$stable_key)
  if (!stable_key %in% names(data) || any(is.na(data[[stable_key]])) || any(data[[stable_key]] == "")) {
    nrc_abort("R stage input has missing stable keys")
  }
  if (!identical(length(unique(as.character(data[[stable_key]]))), as.integer(input$stable_key_count))) {
    nrc_abort("R stage stable-key count does not match the contract")
  }
  list(
    contract = contract,
    data = data,
    contract_path = normalized_contract_path,
    stage_root = stage_root,
    input_path = input_path,
    output_path = output_path
  )
}

nrc_write_stage_result <- function(stage, result) {
  required_result_names <- c("status", "results", "metadata")
  if (!is.list(result) || !all(required_result_names %in% names(result))) {
    nrc_abort("R analysis dispatcher returned an incomplete result")
  }
  if (!result$status %in% c("completed", "skipped", "failed")) {
    nrc_abort("R analysis dispatcher returned an invalid terminal status")
  }
  output_path <- stage$output_path
  temporary_path <- tempfile(pattern = ".result-", tmpdir = dirname(output_path), fileext = ".json")
  payload <- list(
    contract_version = 1L,
    contract_sha256 = stage$contract$contract_sha256,
    input_sha256 = stage$contract$input$sha256,
    status = result$status,
    input_row_count = stage$contract$input$row_count,
    results = result$results,
    metadata = result$metadata
  )
  jsonlite::write_json(payload, temporary_path, auto_unbox = TRUE, null = "null", pretty = FALSE)
  if (file.exists(output_path) && !unlink(output_path)) {
    nrc_abort("Unable to replace an existing R stage result")
  }
  if (!file.rename(temporary_path, output_path)) {
    unlink(temporary_path)
    nrc_abort("Unable to atomically promote the R stage result")
  }
  invisible(output_path)
}
