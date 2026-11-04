
posterior_ui <- function(id) {
  ns <- NS(id)

  tagList(
    div(
      class = "page-shell",

      page_heading(
        "Bayesian Posterior",
        "Posterior inference, uncertainty and MCMC convergence"
      ),

      div(
        class = "two-columns posterior-top",

        panel(
          "Posterior Means and 95% Credible Intervals",
          plotlyOutput(ns("intervals"), height = "350px"),
          icon_name = "chart-line"
        ),

        panel(
          "MCMC Diagnostics",
          uiOutput(ns("diagnostics")),
          icon_name = "wave-square"
        )
      ),

      div(
        class = "two-columns",

        panel(
          "Posterior Summary",
          DTOutput(ns("posterior_table")),
          icon_name = "table"
        ),

        panel(
          "Posterior Group Comparisons",
          DTOutput(ns("comparison_table")),
          icon_name = "scale-balanced"
        )
      ),

      div(
        class = "insight-line",
        icon("lightbulb"),
        "Posterior default probability increases sharply with repayment delay."
      ),

      panel(
        "Interactive Prior Explorer",

        p(
          class = "muted",
          "Experiment with alternative Beta priors. The interactive estimates are exploratory and do not overwrite the reference model."
        ),

        div(
          class = "filter-toolbar",

          selectInput(
            ns("selected_group"),
            "Risk Group",
            choices = risk_order,
            selected = risk_order[1],
            width = "280px"
          ),

          sliderInput(
            ns("alpha"),
            "Prior Alpha",
            min = 1,
            max = 25,
            value = 1,
            step = 1,
            width = "240px"
          ),

          sliderInput(
            ns("beta"),
            "Prior Beta",
            min = 1,
            max = 25,
            value = 1,
            step = 1,
            width = "240px"
          )
        ),

        div(
          class = "interactive-metrics",
          textOutput(ns("prior_insight"))
        ),

        plotlyOutput(ns("density"), height = "390px"),

        icon_name = "sliders"
      )
    )
  )
}

posterior_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {

    output$intervals <- renderPlotly({
      df <- data$posterior

      df$risk_group <- factor(
        df$risk_group,
        levels = rev(risk_order)
      )

      means <- df$posterior_mean * 100

      p <- plot_ly(
        df,
        x = means,
        y = ~risk_group,
        type = "scatter",
        mode = "markers",
        marker = list(
          color = unname(
            risk_colors[as.character(df$risk_group)]
          ),
          size = 14
        ),
        error_x = list(
          type = "data",
          symmetric = FALSE,
          array = df$ci_95_upper * 100 - means,
          arrayminus = means - df$ci_95_lower * 100,
          thickness = 3,
          width = 8
        ),
        text = ~sprintf(
          "%s<br>Posterior PD: %.2f%%<br>95%% CI: %.2f%%–%.2f%%",
          risk_group,
          posterior_mean * 100,
          ci_95_lower * 100,
          ci_95_upper * 100
        ),
        hoverinfo = "text"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 30, b = 65, l = 155, r = 35),
          xaxis = list(
            title = "Posterior Default Probability (%)",
            range = c(0, 80)
          ),
          yaxis = list(title = "")
        )
    })

    output$diagnostics <- renderUI({
      m <- data$mcmc

      div(
        class = "diagnostic-grid",

        div(
          class = "diagnostic-item",
          tags$small("Maximum R-hat"),
          strong(sprintf("%.4f", m$maximum_rhat)),
          span(class = "pill positive", "Near 1.0")
        ),

        div(
          class = "diagnostic-item",
          tags$small("Minimum Bulk ESS"),
          strong(fmt_integer(m$minimum_bulk_ess)),
          span(class = "pill neutral", "Above 400")
        ),

        div(
          class = "diagnostic-item",
          tags$small("Mean Acceptance Rate"),
          strong(fmt_percent(m$mean_acceptance_rate)),
          span(class = "pill neutral", "Metropolis-Hastings")
        ),

        div(
          class = "diagnostic-item",
          tags$small("Sampling Diagnostics"),
          strong(
            if (isTRUE(m$diagnostics_passed)) {
              "PASSED"
            } else {
              "REVIEW"
            }
          ),
          span(
            class = if (isTRUE(m$diagnostics_passed)) {
              "pill positive"
            } else {
              "pill warning"
            },
            "4 independent chains"
          )
        )
      )
    })

    output$posterior_table <- DT::renderDT({
      df <- data$posterior

      small_table(
        data.frame(
          `Risk Group` = df$risk_group,
          Customers = fmt_integer(df$customers),
          Defaults = fmt_integer(df$defaults),
          `Posterior PD` = fmt_percent(df$posterior_mean),
          `95% Credible Interval` = sprintf(
            "%.2f%%–%.2f%%",
            df$ci_95_lower * 100,
            df$ci_95_upper * 100
          ),
          check.names = FALSE
        )
      )
    })

    output$comparison_table <- DT::renderDT({
      df <- data$comparisons

      small_table(
        data.frame(
          Comparison = paste(
            df$group_b,
            ">",
            df$group_a
          ),
          `P(Higher Risk)` = sprintf(
            "%.4f",
            df$p_b_greater_than_a
          ),
          `Mean Difference` = sprintf(
            "%.2f pp",
            df$mean_difference * 100
          ),
          `95% CI` = sprintf(
            "%.2f–%.2f pp",
            df$difference_ci_95_lower * 100,
            df$difference_ci_95_upper * 100
          ),
          check.names = FALSE
        )
      )
    })

    selected <- reactive({
      req(
        input$selected_group,
        input$alpha,
        input$beta
      )

      df <- data$posterior

      df[
        df$risk_group == input$selected_group,
        ,
        drop = FALSE
      ]
    })

    output$prior_insight <- renderText({
      row <- selected()

      n <- row$customers
      y <- row$defaults

      new_mean <- (input$alpha + y) /
        (input$alpha + input$beta + n)

      sprintf(
        paste(
          "Reference Posterior PD: %.3f%%",
          "| New Beta(%d,%d) Prior: %.3f%%",
          "| Change: %+.4f percentage points"
        ),
        row$posterior_mean * 100,
        input$alpha,
        input$beta,
        new_mean * 100,
        (new_mean - row$posterior_mean) * 100
      )
    })

    output$density <- renderPlotly({
      row <- selected()

      y <- row$defaults
      n <- row$customers

      low <- max(0, row$ci_95_lower - 0.08)
      high <- min(1, row$ci_95_upper + 0.08)

      grid <- seq(
        low,
        high,
        length.out = 600
      )

      reference <- dbeta(
        grid,
        y + 1,
        n - y + 1
      )

      updated <- dbeta(
        grid,
        y + input$alpha,
        n - y + input$beta
      )

      df <- data.frame(
        pd = grid * 100,
        reference = reference,
        updated = updated
      )

      p <- plot_ly(df, x = ~pd) %>%
        add_lines(
          y = ~reference,
          name = "Reference Beta(1,1)",
          line = list(
            color = "#64748b",
            width = 3,
            dash = "dash"
          )
        ) %>%
        add_lines(
          y = ~updated,
          name = "Selected Prior",
          line = list(
            color = "#2563eb",
            width = 3
          )
        )

      base_plot(p) %>%
        layout(
          margin = list(t = 35, b = 65, l = 75, r = 25),
          xaxis = list(
            title = "Default Probability (%)"
          ),
          yaxis = list(
            title = "Posterior Density"
          ),
          legend = list(
            orientation = "h",
            x = 0,
            y = 1.15
          )
        )
    })
  })
}