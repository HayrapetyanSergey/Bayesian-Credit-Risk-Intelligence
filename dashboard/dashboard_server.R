
server <- function(input, output, session) {
  data <- load_dashboard_data()

  drawer_open <- reactiveVal(FALSE)

  observeEvent(input$drawer_toggle, {
    drawer_open(!drawer_open())
  })

  navigate <- function(target) {
    bslib::nav_select(
      id = "main_navbar",
      selected = target,
      session = session
    )
    drawer_open(FALSE)
  }

  output$current_page <- renderText({
    current <- input$main_navbar

    if (is.null(current) || !current %in% names(nav_items)) {
      current <- "home"
    }

    nav_items[[current]]
  })

  output$drawer_ui <- renderUI({
    if (!drawer_open()) {
      return(NULL)
    }

    tagList(
      div(
        class = "drawer-backdrop",
        actionButton(
          "drawer_close_backdrop",
          label = NULL,
          class = "backdrop-button"
        )
      ),
      div(
        class = "side-drawer",
        div(
          class = "drawer-heading",
          div(
            icon("chart-line"),
            span("Explore Dashboard")
          ),
          actionButton(
            "drawer_close",
            icon("xmark"),
            class = "drawer-close"
          )
        ),
        div(
          class = "drawer-links",
          lapply(names(nav_items), function(key) {
            actionLink(
              paste0("menu_", key),
              label = tagList(
                icon(nav_icons[[key]]),
                span(nav_items[[key]])
              ),
              class = "drawer-link"
            )
          })
        ),
        div(
          class = "drawer-foot",
          strong("Bayesian Statistics"),
          p("Final research project")
        )
      )
    )
  })

  observeEvent(input$drawer_close, {
    drawer_open(FALSE)
  })

  observeEvent(input$drawer_close_backdrop, {
    drawer_open(FALSE)
  })

  for (name in names(nav_items)) {
    local({
      target <- name

      observeEvent(
        input[[paste0("menu_", target)]],
        navigate(target)
      )
    })
  }

  home_server("home", data, navigate)
  overview_server("overview", data)
  posterior_server("posterior", data)
  prediction_server("prediction", data)
  model_assessment_server("assessment", data)
  ai_assistant_server("assistant", data)
}
