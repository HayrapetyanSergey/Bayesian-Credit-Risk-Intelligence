
home_ui <- function(id) {
  ns <- NS(id)

  info <- list(
    c("graduation-cap", "University", "Yerevan State University"),
    c("layer-group", "Program", "Data Science in Business"),
    c("book-open", "Course", "Bayesian Statistics"),
    c("laptop", "Project type", "Individual Final Project"),
    c("user", "Prepared by", "Sergey Hayrapetyan")
  )

  features <- list(
    c("overview", "chart-column", "Overview",
      "Explore customers, defaults, repayment behavior and the underlying dataset."),
    c("posterior", "chart-line", "Bayesian Posterior",
      "Examine posterior probabilities, credible intervals and MCMC diagnostics."),
    c("prediction", "bullseye", "Prediction",
      "Explore expected future defaults and predictive uncertainty."),
    c("assessment", "shield-halved", "Model Assessment",
      "Review prior sensitivity, validation and model limitations."),
    c("assistant", "robot", "AI Assistant",
      "Ask questions about Bayesian statistics and research findings.")
  )

  tagList(
    div(
      class = "page-shell home-shell",

      tags$section(
        class = "home-intro",

        div(
          class = "home-hero",
          div(class = "eyebrow", "ACADEMIC RESEARCH PROJECT"),
          h1("Bayesian Credit Risk Intelligence"),
          p(
            class = "hero-lead",
            "A probabilistic approach to credit default estimation, uncertainty quantification and predictive decision support."
          ),
          div(
            class = "hero-chips",
            span("Python · Bayesian Inference"),
            span("R Shiny · Interactive Analytics"),
            span("MCMC · Model Diagnostics")
          )
        ),

        div(
          class = "home-details",

          lapply(info, function(item) {
            div(
              class = "detail-card",
              div(
                class = "detail-icon",
                icon(item[1])
              ),
              div(
                class = "detail-copy",
                tags$small(item[2]),
                strong(item[3])
              )
            )
          }),

          div(
            class = "detail-card linkedin-card",
            div(
              class = "detail-icon",
              icon("linkedin", lib = "font-awesome")
            ),
            div(
              class = "detail-copy",
              tags$small("LinkedIn"),
              tags$a(
                href = "https://www.linkedin.com/in/sergey-hayrapetyan-84a16a264/",
                target = "_blank",
                rel = "noopener noreferrer",
                "Sergey Hayrapetyan ",
                icon("arrow-up-right-from-square")
              )
            )
          )
        ),

        div(
          class = "project-description",
          h3(icon("circle-info"), " About the Project"),

          p(
            "This individual final project for Bayesian Statistics, part of the Data Science in Business program at Yerevan State University, investigates next-month credit card default using 30,000 historical customer records."
          ),

          p(
            "The research combines Beta-Binomial inference, analytical posterior estimation, grid approximation, Metropolis-Hastings MCMC, posterior predictive analysis, prior sensitivity and held-out validation."
          ),

          p(
            "The dashboard is an educational research application. It does not represent a production-ready lending decision system."
          )
        )
      ),

      tags$section(
        class = "home-data-section",

        page_heading(
          "Explore the Dataset",
          "Historical observations and Bayesian findings at a glance"
        ),

        div(
          class = "stat-grid",

          div(
            class = "stat-tile",
            tags$small("Borrowers"),
            strong(textOutput(ns("customers"), inline = TRUE))
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
            tags$small("Risk Groups"),
            strong("3")
          )
        ),

        div(
          class = "two-columns",

          panel(
            "Default Distribution",
            plotlyOutput(ns("distribution"), height = "320px"),
            icon_name = "chart-column"
          ),

          panel(
            "Posterior PD by Repayment Status",
            plotlyOutput(ns("risk"), height = "320px"),
            icon_name = "chart-line"
          )
        ),

        div(
          class = "insight-line",
          icon("lightbulb"),
          "The overall observed default rate is 22.12%, while group-level posterior default probabilities differ substantially."
        )
      ),

      tags$section(
        class = "feature-section",

        h2("What this dashboard includes"),

        p(
          "Select any section to explore its interactive analysis."
        ),

        div(
          class = "feature-grid",

          lapply(features, function(item) {
            div(
              class = "feature-card",

              div(
                class = "feature-top",
                div(
                  class = "feature-icon",
                  icon(item[2])
                ),

                actionLink(
                  ns(paste0("jump_", item[1])),
                  icon("arrow-right"),
                  class = "feature-action"
                )
              ),

              h3(item[3]),
              p(item[4])
            )
          })
        )
      )
    )
  )
}

home_server <- function(id, data, navigate) {
  moduleServer(id, function(input, output, session) {

    for (target in c(
      "overview",
      "posterior",
      "prediction",
      "assessment",
      "assistant"
    )) {
      local({
        key <- target

        observeEvent(
          input[[paste0("jump_", key)]],
          navigate(key)
        )
      })
    }

    output$customers <- renderText({
      fmt_integer(data$metrics$customers)
    })

    output$defaults <- renderText({
      fmt_integer(data$metrics$defaults)
    })

    output$rate <- renderText({
      fmt_percent(data$metrics$observed_rate)
    })

    output$distribution <- renderPlotly({
      df <- data.frame(
        status = c("Default", "Non-default"),
        customers = c(
          data$metrics$defaults,
          data$metrics$customers - data$metrics$defaults
        )
      )

      p <- plot_ly(
        df,
        x = ~status,
        y = ~customers,
        type = "bar",
        marker = list(
          color = c("#ef4444", "#18345b")
        ),
        text = ~fmt_integer(customers),
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
      df <- data$posterior

      df$risk_group <- factor(
        df$risk_group,
        levels = rev(risk_order)
      )

      df$pd_pct <- df$posterior_mean * 100

      p <- plot_ly(
        df,
        x = ~pd_pct,
        y = ~risk_group,
        type = "bar",
        orientation = "h",
        marker = list(
          color = unname(
            risk_colors[as.character(df$risk_group)]
          )
        ),
        text = ~sprintf("%.2f%%", pd_pct),
        textposition = "outside"
      )

      base_plot(p) %>%
        layout(
          showlegend = FALSE,
          margin = list(t = 25, b = 55, l = 150, r = 50),
          xaxis = list(
            title = "Posterior Default Probability (%)",
            range = c(0, 82)
          ),
          yaxis = list(title = "")
        )
    })
  })
}
