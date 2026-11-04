
model_assessment_ui <- function(id) {
  ns <- NS(id)

  tagList(
    div(
      class = "page-shell",

      page_heading(
        "Model Assessment",
        "Robustness, held-out validation and transparent model limitations"
      ),

      div(
        class = "stat-grid",

        div(
          class = "stat-tile",
          tags$small("Maximum Prior Impact"),
          strong(textOutput(ns("prior_impact"), inline = TRUE))
        ),

        div(
          class = "stat-tile orange",
          tags$small("Groups Inside 95% Intervals"),
          strong(textOutput(ns("coverage"), inline = TRUE))
        ),

        div(
          class = "stat-tile blue",
          tags$small("Weighted Brier Score"),
          strong(textOutput(ns("brier"), inline = TRUE))
        ),

        div(
          class = "stat-tile green",
          tags$small("Weighted Log Loss"),
          strong(textOutput(ns("logloss"), inline = TRUE))
        )
      ),

      div(
        class = "two-columns",

        panel(
          "Prior Sensitivity Analysis",
          plotlyOutput(ns("prior_plot"), height = "345px"),
          icon_name = "layer-group"
        ),

        panel(
          "Held-out: Observed vs Predicted",
          plotlyOutput(ns("validation_plot"), height = "345px"),
          icon_name = "shield-halved"
        )
      ),

      div(
        class = "two-columns",

        panel(
          "Prior Sensitivity Estimates",
          DTOutput(ns("prior_table")),
          icon_name = "table"
        ),

        panel(
          "Held-out Validation by Group",
          DTOutput(ns("validation_table")),
          icon_name = "table"
        )
      ),

      panel(
        "Interactive Calibration Inspection",

        p(
          class = "muted",
          "Select a risk group to compare its predicted default rate with the observed rate in the test set."
        ),

        selectInput(
          ns("selected_group"),
          "Risk Group",
          choices = risk_order,
          selected = risk_order[1],
          width = "300px"
        ),

        uiOutput(ns("calibration_detail")),

        plotlyOutput(
          ns("calibration_chart"),
          height = "310px"
        ),

        icon_name = "magnifying-glass-chart"
      ),

      div(
        class = "insight-line warning-line",
        icon("circle-exclamation"),
        "The no-delay group failed its held-out predictive interval check. The model is informative but not fully validated for future lending decisions."
      )
    )
  )
}

model_assessment_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {

    prior_spread <- data$sensitivity %>%
      group_by(risk_group) %>%
      summarise(
        delta = max(posterior_mean) -
          min(posterior_mean),
        .groups = "drop"
      )

    output$prior_impact <- renderText({
      sprintf(
        "%.3f pp",
        100 * max(prior_spread$delta)
      )
    })

    output$coverage <- renderText({
      sprintf(
        "%d / %d",
        sum(data$validation$coverage),
        nrow(data$validation)
      )
    })

    output$brier <- renderText({
      sprintf(
        "%.4f",
        data$metrics$weighted_brier
      )
    })

    output$logloss <- renderText({
      sprintf(
        "%.4f",
        data$metrics$weighted_log_loss
      )
    })

    output$prior_plot <- renderPlotly({
      df <- data$sensitivity

      df$risk_group <- factor(
        df$risk_group,
        levels = risk_order
      )

      df$posterior_pct <- df$posterior_mean * 100

      p <- plot_ly(
        df,
        x = ~risk_group,
        y = ~posterior_pct,
        type = "bar",
        color = ~prior,
        colors = c(
          "#2563eb",
          "#f59e0b",
          "#8b5cf6"
        )
      )

      base_plot(p) %>%
        layout(
          barmode = "group",
          margin = list(t = 60, b = 75, l = 70, r = 20),
          yaxis = list(
            title = "Posterior PD (%)",
            rangemode = "tozero"
          ),
          xaxis = list(title = ""),
          legend = list(
            orientation = "h",
            x = 0,
            y = 1.3
          )
        )
    })

    output$validation_plot <- renderPlotly({
      df <- data$validation

      df$risk_group <- factor(
        df$risk_group,
        levels = rev(risk_order)
      )

      expected <- df$expected_test_defaults

      p <- plot_ly(
        df,
        x = expected,
        y = ~risk_group,
        type = "scatter",
        mode = "markers",
        name = "Predicted",
        marker = list(
          color = "#2563eb",
          size = 13,
          symbol = "diamond"
        ),
        error_x = list(
          type = "data",
          symmetric = FALSE,
          array = df$predictive_upper_95 - expected,
          arrayminus = expected -
            df$predictive_lower_95,
          thickness = 4,
          width = 9,
          color = "#8eb7fa"
        )
      ) %>%
        add_markers(
          x = ~test_defaults,
          y = ~risk_group,
          name = "Observed",
          marker = list(
            color = "#ef4444",
            size = 13
          ),
          text = ~sprintf(
            "Observed: %d<br>Covered: %s",
            test_defaults,
            ifelse(coverage, "Yes", "No")
          ),
          hoverinfo = "text"
        )

      base_plot(p) %>%
        layout(
          margin = list(t = 55, b = 65, l = 160, r = 25),
          xaxis = list(
            title = "Defaults in Test Set",
            rangemode = "tozero"
          ),
          yaxis = list(title = ""),
          legend = list(
            orientation = "h",
            x = 0,
            y = 1.25
          )
        )
    })

    output$prior_table <- DT::renderDT({
      df <- data$sensitivity

      small_table(
        data.frame(
          Prior = df$prior,
          `Risk Group` = df$risk_group,
          `Train N` = fmt_integer(
            df$train_customers
          ),
          `Posterior PD` = sprintf(
            "%.3f%%",
            df$posterior_mean * 100
          ),
          check.names = FALSE
        ),
        page_length = 9
      )
    })

    output$validation_table <- DT::renderDT({
      df <- data$validation

      small_table(
        data.frame(
          `Risk Group` = df$risk_group,
          `Test N` = fmt_integer(
            df$test_customers
          ),
          Observed = df$test_defaults,
          Expected = round(
            df$expected_test_defaults,
            1
          ),
          `95% Interval` = sprintf(
            "%d–%d",
            df$predictive_lower_95,
            df$predictive_upper_95
          ),
          Covered = ifelse(
            df$coverage,
            "Yes",
            "No"
          ),
          check.names = FALSE
        )
      )
    })

    selected_group <- reactive({
      req(input$selected_group)

      df <- data$validation

      df[
        df$risk_group == input$selected_group,
        ,
        drop = FALSE
      ]
    })

    output$calibration_detail <- renderUI({
      df <- selected_group()

      div(
        class = "calibration-summary",

        div(
          tags$small("Posterior Train PD"),
          strong(fmt_percent(df$posterior_mean_pd))
        ),

        div(
          tags$small("Observed Test Rate"),
          strong(fmt_percent(df$observed_test_rate))
        ),

        div(
          tags$small("95% Interval Check"),
          strong(
            if (df$coverage) {
              "Inside Interval"
            } else {
              "Outside Interval"
            }
          )
        )
      )
    })

    output$calibration_chart <- renderPlotly({
      df <- selected_group()

      values <- c(
        df$posterior_mean_pd,
        df$observed_test_rate
      ) * 100

      p <- plot_ly(
        x = c(
          "Predicted Rate",
          "Observed Test Rate"
        ),
        y = values,
        type = "bar",
        marker = list(
          color = c(
            "#2563eb",
            if (df$coverage) "#10b981" else "#ef4444"
          )
        ),
        text = sprintf("%.2f%%", values),
        textposition = "outside"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 40, b = 55, l = 65, r = 20),
          yaxis = list(
            title = "Default Rate (%)",
            range = c(0, max(values) * 1.3)
          ),
          xaxis = list(title = "")
        )
    })
  })
}