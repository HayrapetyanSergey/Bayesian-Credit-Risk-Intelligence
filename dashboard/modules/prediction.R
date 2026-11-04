
prediction_ui <- function(id) {
  ns <- NS(id)

  tagList(
    div(
      class = "page-shell",

      page_heading(
        "Prediction",
        "Explore posterior predictive scenarios and download risk forecasts"
      ),

      div(
        class = "prediction-layout",

        panel(
          "Scenario Selection",

          selectInput(
            ns("n"),
            "Future Customers per Risk Group",
            choices = c(
              "100" = 100,
              "500" = 500,
              "1,000" = 1000
            ),
            selected = 1000
          ),

          div(
            class = "scenario-details",
            uiOutput(ns("scenario_info"))
          ),

          downloadButton(
            ns("download_scenario"),
            "Download This Scenario",
            class = "btn-primary full-download"
          ),

          icon_name = "sliders"
        ),

        panel(
          "Expected Defaults and 95% Predictive Intervals",
          plotlyOutput(
            ns("prediction_plot"),
            height = "350px"
          ),
          icon_name = "bullseye"
        )
      ),

      div(
        class = "two-columns compact-cards",

        div(
          class = "insight-card red",
          uiOutput(ns("high_risk"))
        ),

        div(
          class = "insight-card orange",
          uiOutput(ns("threshold_probability"))
        )
      ),

      panel(
        "Posterior Predictive Summary",

        DTOutput(ns("prediction_table")),

        div(
          class = "panel-footer",

          downloadButton(
            ns("download_all_scenarios"),
            "Download All Scenarios CSV",
            class = "btn-outline-primary"
          )
        ),

        icon_name = "table"
      ),

      div(
        class = "insight-line",
        icon("circle-info"),
        "Predictions assume future borrowers are comparable to historical customers within each repayment group."
      )
    )
  )
}

prediction_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {

    scenario <- reactive({
      req(input$n)

      df <- data$predictive

      df[
        as.integer(df$future_customers) ==
          as.integer(input$n),
        ,
        drop = FALSE
      ]
    })

    output$scenario_info <- renderUI({
      n <- as.integer(input$n)

      tagList(
        div(
          class = "scenario-stat",
          tags$small("Customers per Group"),
          strong(fmt_integer(n))
        ),

        div(
          class = "scenario-stat",
          tags$small("Total across 3 Groups"),
          strong(fmt_integer(n * 3))
        ),

        div(
          class = "scenario-stat",
          tags$small("Predictive Interval"),
          strong("95%")
        )
      )
    })

    output$prediction_plot <- renderPlotly({
      df <- scenario()

      df$risk_group <- factor(
        df$risk_group,
        levels = rev(risk_order)
      )

      p <- plot_ly(
        df,
        x = ~expected_defaults,
        y = ~risk_group,
        type = "scatter",
        mode = "markers",
        marker = list(
          color = unname(
            risk_colors[as.character(df$risk_group)]
          ),
          size = 15
        ),
        error_x = list(
          type = "data",
          symmetric = FALSE,
          array = df$predictive_upper_95 -
            df$expected_defaults,
          arrayminus = df$expected_defaults -
            df$predictive_lower_95,
          thickness = 3,
          width = 9
        ),
        text = ~sprintf(
          "%s<br>Expected: %.2f<br>95%% Interval: %d–%d",
          risk_group,
          expected_defaults,
          predictive_lower_95,
          predictive_upper_95
        ),
        hoverinfo = "text"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 30, b = 65, l = 160, r = 35),
          xaxis = list(
            title = "Expected Number of Defaults",
            rangemode = "tozero"
          ),
          yaxis = list(title = "")
        )
    })

    output$high_risk <- renderUI({
      df <- scenario()

      row <- df[
        which.max(df$expected_defaults),
        ,
        drop = FALSE
      ]

      tagList(
        h4(
          icon("triangle-exclamation"),
          " Highest Expected Defaults"
        ),

        p(
          sprintf(
            "%s: %.1f expected defaults. 95%% interval: %d–%d.",
            row$risk_group,
            row$expected_defaults,
            row$predictive_lower_95,
            row$predictive_upper_95
          )
        )
      )
    })

    output$threshold_probability <- renderUI({
      df <- scenario()

      row <- df[
        df$risk_group == "1-month delay",
        ,
        drop = FALSE
      ]

      tagList(
        h4(
          icon("chart-line"),
          " Probability of More Than 30% Defaults"
        ),

        p(
          sprintf(
            "One-month delay group: %.2f%% posterior predictive probability for %s customers.",
            100 * row$prob_more_than_30pct_default,
            fmt_integer(row$future_customers)
          )
        )
      )
    })

    output$prediction_table <- DT::renderDT({
      df <- scenario()

      small_table(
        data.frame(
          `Risk Group` = df$risk_group,
          `Future Customers` = fmt_integer(
            df$future_customers
          ),
          `Posterior PD` = fmt_percent(
            df$posterior_mean_pd
          ),
          `Expected Defaults` = sprintf(
            "%.2f",
            df$expected_defaults
          ),
          `95% Predictive Interval` = sprintf(
            "%d–%d",
            df$predictive_lower_95,
            df$predictive_upper_95
          ),
          `P(>30% Defaults)` = fmt_percent(
            df$prob_more_than_30pct_default
          ),
          check.names = FALSE
        )
      )
    })

    output$download_scenario <- downloadHandler(
      filename = function() {
        sprintf(
          "predictive_scenario_%s.csv",
          input$n
        )
      },

      content = function(file) {
        write.csv(
          scenario(),
          file,
          row.names = FALSE
        )
      }
    )

    output$download_all_scenarios <- downloadHandler(
      filename = function() {
        "posterior_predictive_all_scenarios.csv"
      },

      content = function(file) {
        write.csv(
          data$predictive,
          file,
          row.names = FALSE
        )
      }
    )
  })
}