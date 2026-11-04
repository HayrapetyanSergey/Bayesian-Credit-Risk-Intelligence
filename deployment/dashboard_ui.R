
nav_items <- c(
  home = "Home",
  overview = "Overview",
  posterior = "Bayesian Posterior",
  prediction = "Prediction",
  assessment = "Model Assessment",
  assistant = "AI Assistant"
)

nav_icons <- c(
  home = "house",
  overview = "chart-column",
  posterior = "chart-line",
  prediction = "bullseye",
  assessment = "shield-halved",
  assistant = "robot"
)

ui <- bslib::page_navbar(
  title = div(
    class = "brand-layout",
    actionButton(
      "drawer_toggle",
      icon("bars"),
      class = "hamburger",
      title = "Open navigation"
    ),
    div(
      class = "brand-name",
      icon("chart-line"),
      span("Bayesian Credit Risk Intelligence")
    ),
    span(
      class = "current-page",
      textOutput("current_page", inline = TRUE)
    )
  ),
  id = "main_navbar",
  selected = "home",
  fillable = FALSE,
  theme = bslib::bs_theme(
    version = 5,
    primary = "#2563eb",
    base_font = bslib::font_google("Inter")
  ),
  header = tagList(
    tags$head(
  includeCSS("www/custom.css"),
  tags$script(HTML("
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      },
      startup: {
        typeset: false
      }
    };
  ")),
  tags$script(
    src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
  ),
  tags$script(HTML("
    function renderAgentMath() {
      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise(
          Array.from(document.querySelectorAll('.agent-answer'))
        ).catch(function(error) {
          console.error(error);
        });
      }
    }

    $(document).on('shiny:value', function(event) {
      if (event.name && event.name.indexOf('conversation') !== -1) {
        setTimeout(renderAgentMath, 50);
      }
    });
  "))
),
    uiOutput("drawer_ui")
  ),

  bslib::nav_panel(
    "Home",
    value = "home",
    home_ui("home")
  ),

  bslib::nav_panel(
    "Overview",
    value = "overview",
    overview_ui("overview")
  ),

  bslib::nav_panel(
    "Bayesian Posterior",
    value = "posterior",
    posterior_ui("posterior")
  ),

  bslib::nav_panel(
    "Prediction",
    value = "prediction",
    prediction_ui("prediction")
  ),

  bslib::nav_panel(
    "Model Assessment",
    value = "assessment",
    model_assessment_ui("assessment")
  ),

  bslib::nav_panel(
    "AI Assistant",
    value = "assistant",
    ai_assistant_ui("assistant")
  )
)
