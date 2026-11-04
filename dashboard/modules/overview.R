
overview_ui <- function(id) {
  ns <- NS(id)

  tagList(
    div(
      class = "page-shell",

      page_heading(
        "Overview",
        "Explore historical default patterns and inspect the underlying observations"
      ),

      div(
        class = "stat-grid",

        div(
          class = "stat-tile",
          tags$small("Total Customers"),
          strong(textOutput(ns("total"), inline = TRUE))
        ),

        div(
          class = "stat-tile red",
          tags$small("Observed Defaults"),
          strong(textOutput(ns("defaults"), inline = TRUE))
        ),

        div(
          class = "stat-tile blue",
          tags$small("Observed Default Rate"),
          strong(textOutput(ns("rate"), inline = TRUE))
        ),

        div(
          class = "stat-tile green",
          tags$small("Average Credit Limit (NT$)"),
          strong(textOutput(ns("limit"), inline = TRUE))
        )
      ),

      div(
        class = "two-columns",

        panel(
          "Default Distribution",
          plotlyOutput(ns("distribution"), height = "330px"),
          icon_name = "chart-column"
        ),

        panel(
          "Repayment Delay Risk Groups",
          plotlyOutput(ns("risk"), height = "330px"),
          icon_name = "layer-group"
        ),

        panel(
          "Default Rate by Age",
          plotlyOutput(ns("age"), height = "330px"),
          icon_name = "users"
        ),

        panel(
          "Default Rate by Credit Limit",
          plotlyOutput(ns("limits"), height = "330px"),
          icon_name = "credit-card"
        )
      ),

      panel(
        "Explore & Download the Dataset",

        p(
          class = "muted",
          "Filter customers by default outcome and repayment group. Search within the table, inspect observations and download results."
        ),

        div(
          class = "filter-toolbar",

          selectInput(
            ns("status"),
            "Default Outcome",
            choices = c(
              "All" = "all",
              "Non-default" = "0",
              "Default" = "1"
            ),
            width = "220px"
          ),

          selectInput(
            ns("group"),
            "Risk Group",
            choices = c("All", risk_order),
            selected = "All",
            width = "250px"
          ),

          downloadButton(
            ns("download_all"),
            "Download Full Dataset",
            class = "btn-outline-primary"
          ),

          downloadButton(
            ns("download_filtered"),
            "Download Filtered Dataset",
            class = "btn-primary"
          )
        ),

        div(
          class = "filter-note",
          textOutput(ns("shown_count"))
        ),

        DTOutput(ns("dataset_table")),

        icon_name = "table"
      )
    )
  )
}

overview_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {

    output$total <- renderText({
      fmt_integer(data$metrics$customers)
    })

    output$defaults <- renderText({
      fmt_integer(data$metrics$defaults)
    })

    output$rate <- renderText({
      fmt_percent(data$metrics$observed_rate)
    })

    output$limit <- renderText({
      paste0("NT$", fmt_integer(data$metrics$mean_limit))
    })

    output$distribution <- renderPlotly({
      df <- data.frame(
        status = c("Default", "Non-default"),
        n = c(
          data$metrics$defaults,
          data$metrics$customers - data$metrics$defaults
        )
      )

      p <- plot_ly(
        df,
        x = ~status,
        y = ~n,
        type = "bar",
        marker = list(
          color = c("#ef4444", "#18345b")
        ),
        text = ~fmt_integer(n),
        textposition = "outside"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 40, b = 45, l = 65, r = 20),
          yaxis = list(title = "Customers"),
          xaxis = list(title = "")
        )
    })

    output$risk <- renderPlotly({
      df <- data$risk

      df$risk_group <- factor(
        df$risk_group,
        levels = risk_order
      )

      p <- plot_ly(
        df,
        x = ~risk_group,
        y = ~default_rate_pct,
        type = "bar",
        marker = list(
          color = unname(
            risk_colors[as.character(df$risk_group)]
          )
        ),
        text = ~sprintf("%.2f%%", default_rate_pct),
        textposition = "outside"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 40, b = 75, l = 65, r = 20),
          yaxis = list(
            title = "Default Rate (%)",
            range = c(0, 80)
          ),
          xaxis = list(title = "")
        )
    })

    output$age <- renderPlotly({
      df <- data$age

      p <- plot_ly(
        df,
        x = ~age_group,
        y = ~default_rate_pct,
        type = "bar",
        marker = list(color = "#2563eb"),
        text = ~sprintf("%.2f%%", default_rate_pct),
        textposition = "outside"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 40, b = 45, l = 65, r = 20),
          yaxis = list(
            title = "Default Rate (%)",
            range = c(0, 35)
          ),
          xaxis = list(title = "")
        )
    })

    output$limits <- renderPlotly({
      df <- data$limits

      df$credit_limit_group <- factor(
        df$credit_limit_group,
        levels = c(
          "<=50K",
          "50K-100K",
          "100K-200K",
          "200K-500K",
          ">500K"
        )
      )

      p <- plot_ly(
        df,
        x = ~credit_limit_group,
        y = ~default_rate_pct,
        type = "bar",
        marker = list(color = "#0d9488"),
        text = ~sprintf("%.2f%%", default_rate_pct),
        textposition = "outside"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 40, b = 65, l = 65, r = 20),
          yaxis = list(
            title = "Default Rate (%)",
            range = c(0, 40)
          ),
          xaxis = list(title = "")
        )
    })

    filtered <- reactive({
      df <- data$credit

      if (!is.null(input$status) && input$status != "all") {
        df <- df[
          df$default == as.integer(input$status),
          ,
          drop = FALSE
        ]
      }

      if (!is.null(input$group) && input$group != "All") {
        df <- df[
          as.character(df$risk_group) == input$group,
          ,
          drop = FALSE
        ]
      }

      df
    })

    output$shown_count <- renderText({
      paste(
        fmt_integer(nrow(filtered())),
        "rows match the selected dropdown filters."
      )
    })

    table_data <- reactive({
      df <- filtered()

      preferred <- c(
        "id",
        "default",
        "risk_group",
        "age",
        "limit_bal",
        "sex",
        "education",
        "marriage",
        "pay_1",
        "bill_amt1",
        "pay_amt1"
      )

      df[, c(
        intersect(preferred, names(df)),
        setdiff(names(df), preferred)
      ), drop = FALSE]
    })

    output$dataset_table <- DT::renderDT({
      DT::datatable(
        table_data(),
        filter = "top",
        rownames = FALSE,
        class = "compact stripe hover",
        options = list(
          pageLength = 10,
          lengthMenu = c(10, 25, 50),
          scrollX = TRUE,
          autoWidth = TRUE,
          searchHighlight = TRUE,
          processing = TRUE
        )
      )
    }, server = TRUE)

    output$download_all <- downloadHandler(
      filename = function() {
        "credit_default_full.csv"
      },
      content = function(file) {
        write.csv(
          data$credit,
          file,
          row.names = FALSE
        )
      }
    )

    output$download_filtered <- downloadHandler(
      filename = function() {
        "credit_default_filtered.csv"
      },
      content = function(file) {
        df <- table_data()
        rows <- input$dataset_table_rows_all

        if (!is.null(rows)) {
          df <- df[rows, , drop = FALSE]
        }

        write.csv(df, file, row.names = FALSE)
      }
    )
  })
}