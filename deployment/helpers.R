
project_root <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

risk_order <- c(
  "No reported delay",
  "1-month delay",
  "2+ months delay"
)

risk_colors <- c(
  "No reported delay" = "#2563eb",
  "1-month delay" = "#f59e0b",
  "2+ months delay" = "#ef4444"
)

read_project_csv <- function(filename) {
  path <- file.path(
    project_root,
    "results",
    "tables",
    filename
  )

  if (!file.exists(path)) {
    stop(paste("Required result missing:", path))
  }

  readr::read_csv(path, show_col_types = FALSE)
}

read_project_json <- function(filename) {
  path <- file.path(
    project_root,
    "results",
    "tables",
    filename
  )

  if (!file.exists(path)) {
    stop(paste("Required report missing:", path))
  }

  jsonlite::fromJSON(path)
}

load_dashboard_data <- function() {
  path <- file.path(
    project_root,
    "data",
    "processed",
    "credit_with_groups.csv"
  )

  if (!file.exists(path)) {
    stop("Run src/exploratory_analysis.py first")
  }

  credit <- readr::read_csv(
    path,
    show_col_types = FALSE
  )

  required <- c(
    "id",
    "default",
    "risk_group",
    "limit_bal",
    "age"
  )

  if (!all(required %in% names(credit))) {
    stop("Processed dataset is missing required columns")
  }

  posterior <- read_project_csv(
    "bayesian_posterior_summary.csv"
  )

  comparisons <- read_project_csv(
    "posterior_group_comparisons.csv"
  )

  predictive <- read_project_csv(
    "posterior_predictive_summary.csv"
  )

  sensitivity <- read_project_csv(
    "prior_sensitivity_summary.csv"
  )

  validation <- read_project_csv(
    "heldout_validation_summary.csv"
  )

  risk <- read_project_csv(
    "risk_group_summary.csv"
  )

  age <- read_project_csv(
    "age_group_summary.csv"
  )

  limits <- read_project_csv(
    "credit_limit_group_summary.csv"
  )

  mcmc <- read_project_json(
    "mcmc_report.json"
  )

  credit$risk_group <- factor(
    credit$risk_group,
    levels = risk_order
  )

  list(
    credit = credit,
    posterior = posterior,
    comparisons = comparisons,
    predictive = predictive,
    sensitivity = sensitivity,
    validation = validation,
    risk = risk,
    age = age,
    limits = limits,
    mcmc = mcmc,

    metrics = list(
      customers = nrow(credit),
      defaults = sum(credit$default),
      observed_rate = mean(credit$default),
      mean_limit = mean(credit$limit_bal),

      weighted_brier = weighted.mean(
        validation$brier_score,
        validation$test_customers
      ),

      weighted_log_loss = weighted.mean(
        validation$log_loss,
        validation$test_customers
      )
    )
  )
}

page_heading <- function(title, subtitle) {
  div(
    class = "page-heading",
    h1(title),
    p(subtitle),
    div(class = "heading-line")
  )
}

panel <- function(title, ..., icon_name = NULL, class = "") {
  div(
    class = paste("panel-surface", class),
    div(
      class = "panel-title",
      if (!is.null(icon_name)) icon(icon_name),
      span(title)
    ),
    div(
      class = "panel-body",
      ...
    )
  )
}

fmt_percent <- function(x, digits = 2) {
  sprintf(
    paste0("%.", digits, "f%%"),
    100 * x
  )
}

fmt_integer <- function(x) {
  format(
    round(x),
    big.mark = ",",
    trim = TRUE,
    scientific = FALSE
  )
}

base_plot <- function(p) {
  p %>%
    plotly::layout(
      font = list(
        family = "Arial, sans-serif",
        color = "#253858",
        size = 12
      ),
      paper_bgcolor = "rgba(0,0,0,0)",
      plot_bgcolor = "rgba(0,0,0,0)",
      hoverlabel = list(
        bgcolor = "#0d2451",
        font = list(color = "#ffffff")
      )
    ) %>%
    plotly::config(
      displaylogo = FALSE,
      responsive = TRUE
    )
}

small_table <- function(df, filter = "none", page_length = 8) {
  DT::datatable(
    df,
    rownames = FALSE,
    filter = filter,
    class = "compact stripe hover",
    options = list(
      dom = "tip",
      pageLength = page_length,
      scrollX = TRUE,
      autoWidth = TRUE,
      ordering = TRUE,
      language = list(
        emptyTable = "No observations match the filters"
      )
    )
  )
}