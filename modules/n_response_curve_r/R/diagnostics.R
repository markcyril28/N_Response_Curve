nrc_package_versions <- function() {
  packages <- c("arrow", "broom", "digest", "glmmTMB", "jsonlite", "lme4")
  versions <- lapply(packages, function(package_name) {
    if (!requireNamespace(package_name, quietly = TRUE)) {
      return(NA_character_)
    }
    as.character(utils::packageVersion(package_name))
  })
  stats::setNames(unlist(versions, use.names = FALSE), packages)
}

nrc_model_diagnostics <- function(model, warnings = character()) {
  singular <- FALSE
  if (inherits(model, "merMod")) {
    singular <- lme4::isSingular(model, tol = 1e-4)
  }
  list(
    converged = TRUE,
    singular = singular,
    boundary_fit = singular,
    residual_df = tryCatch(stats::df.residual(model), error = function(error) NA_integer_),
    warning_messages = as.list(unique(warnings)),
    contrast_coding = "R defaults; explicit contrast specification required for interpretation",
    package_versions = as.list(nrc_package_versions()),
    r_version = R.version.string
  )
}

nrc_tidy_model <- function(model) {
  if (!requireNamespace("broom", quietly = TRUE)) {
    return(list())
  }
  tidy <- tryCatch(
    broom::tidy(model, conf.int = TRUE),
    error = function(error) broom::tidy(model)
  )
  nrc_frame_to_rows(as.data.frame(tidy))
}
