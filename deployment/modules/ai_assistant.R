
ai_assistant_ui <- function(id) {
  ns <- NS(id)

  questions <- c(
    "What is the overall default rate?",
    "Which group is the riskiest?",
    "What is a 95% credible interval?",
    "Did MCMC converge?",
    "How many defaults are expected for 1000 customers?",
    "Was the model sensitive to priors?",
    "What happened in held-out validation?"
  )

  capabilities <- list(
    c("database", "Dataset Facts"),
    c("chart-line", "Posterior Probabilities"),
    c("arrows-left-right", "Credible Intervals"),
    c("wave-square", "MCMC Diagnostics"),
    c("bullseye", "Predictions"),
    c("shield-halved", "Validation")
  )

  tagList(
    div(
      class = "page-shell",

      page_heading(
        "AI Assistant",
        "Explore Bayesian credit risk with a Gemini-powered research agent"
      ),

      div(
        class = "assistant-layout",

        panel(
          "Ask the Bayesian Assistant",

          div(
            class = "assistant-intro",
            icon("circle-info"),
            "Connected to Python statistical tools and Google Gemini. Project-specific numbers are retrieved from the research data."
          ),

          div(
            class = "conversation",
            uiOutput(ns("conversation"))
          ),

          div(
            class = "ask-row",

            textInput(
              ns("question"),
              label = NULL,
              placeholder = "Ask a question about the analysis...",
              width = "100%"
            ),

            actionButton(
              ns("ask"),
              "Ask",
              icon = icon("paper-plane"),
              class = "btn-primary"
            )
          ),

          div(
            class = "assistant-controls",

            actionButton(
              ns("clear"),
              "Clear Conversation",
              icon = icon("trash"),
              class = "btn-outline-secondary"
            ),

            textOutput(ns("status"), inline = TRUE)
          ),

          icon_name = "comments"
        ),

        div(
          class = "assistant-right",

          panel(
            "Suggested Questions",

            div(
              class = "question-stack",

              lapply(seq_along(questions), function(i) {
                actionLink(
                  ns(paste0("suggest_", i)),
                  label = span(questions[i]),
                  class = "suggested-question"
                )
              })
            ),

            icon_name = "circle-question"
          ),

          panel(
            "What the Assistant Can Answer",

            div(
              class = "capability-grid",

              lapply(seq_along(capabilities), function(i) {
                div(
                  class = paste0("capability-chip cap-", i),
                  icon(capabilities[[i]][1]),
                  strong(capabilities[[i]][2])
                )
              })
            ),

            icon_name = "layer-group"
          )
        )
      )
    )
  )
}

ai_assistant_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {

    api_url <- Sys.getenv(
      "BAYESIAN_AGENT_API_URL",
      "http://127.0.0.1:8000/chat"
    )

    questions <- c(
      "What is the overall default rate?",
      "Which group is the riskiest?",
      "What is a 95% credible interval?",
      "Did MCMC converge?",
      "How many defaults are expected for 1000 customers?",
      "Was the model sensitive to priors?",
      "What happened in held-out validation?"
    )

    history <- reactiveVal(list())
    busy <- reactiveVal(FALSE)
    connection_status <- reactiveVal("Ready")

    for (i in seq_along(questions)) {
      local({
        j <- i

        observeEvent(
          input[[paste0("suggest_", j)]],
          {
            updateTextInput(
              session,
              "question",
              value = questions[j]
            )
          }
        )
      })
    }

    send_question <- function(question) {
      messages <- history()

      previous_history <- list()

      if (length(messages) > 0) {
        for (item in tail(messages, 3)) {
          if (isTRUE(item$success)) {
            previous_history <- append(
              previous_history,
              list(
                list(
                  role = "user",
                  content = item$question
                ),
                list(
                  role = "assistant",
                  content = item$response
                )
              )
            )
          }
        }
      }

      request <- httr2::request(api_url)

      request <- httr2::req_method(
        request,
        "POST"
      )

      request <- httr2::req_body_json(
        request,
        list(
          question = question,
          history = previous_history
        ),
        auto_unbox = TRUE
      )

      request <- httr2::req_timeout(
        request,
        seconds = 240
      )

      response <- httr2::req_perform(request)

      result <- httr2::resp_body_json(
        response,
        simplifyVector = FALSE
      )

      if (is.null(result$answer)) {
        stop("The AI backend returned no answer.")
      }

      result
    }

    observeEvent(input$ask, {
      question <- trimws(input$question)

      if (!nzchar(question) || busy()) {
        return()
      }

      busy(TRUE)
      connection_status("Contacting Gemini Agent...")

      result <- tryCatch(
        send_question(question),
        error = function(error) {
          list(
            answer = paste(
              "The AI backend could not complete this request.",
              "Make sure the FastAPI server is running.",
              conditionMessage(error)
            ),
            model = "Unavailable",
            tools_used = list(),
            failed = TRUE
          )
        }
      )

      succeeded <- !isTRUE(result$failed)

      history(
        append(
          history(),
          list(
            list(
              question = question,
              response = result$answer,
              model = result$model,
              tools_used = result$tools_used,
              success = succeeded
            )
          )
        )
      )

      updateTextInput(
        session,
        "question",
        value = ""
      )

      connection_status(
        if (succeeded) {
          paste("Connected ·", result$model)
        } else {
          "Backend unavailable"
        }
      )

      busy(FALSE)
    })

    observeEvent(input$clear, {
      history(list())
      connection_status("Conversation cleared")
    })

    output$status <- renderText({
      connection_status()
    })

    output$conversation <- renderUI({
      messages <- history()

      if (!length(messages)) {
        return(
          div(
            class = "chat-empty",
            icon("comments"),
            h4("Start a Conversation"),
            p(
              "Choose a suggested question or ask your own."
            )
          )
        )
      }

      tagList(
        lapply(messages, function(entry) {

          tools <- entry$tools_used

          tool_names <- character(0)

          if (length(tools) > 0) {
            tool_names <- vapply(
              tools,
              function(item) item$name,
              character(1)
            )
          }

          tagList(
            div(
              class = "message user-message",

              div(
                class = "message-role",
                icon("user"),
                " Your Question"
              ),

              p(entry$question)
            ),

            div(
              class = "message assistant-message",

              div(
                class = "message-role",
                icon("robot"),
                " Bayesian AI Agent"
              ),

              div(
                class = "agent-answer",
                HTML(
                  commonmark::markdown_html(
                    entry$response,
                    extensions = c("table", "strikethrough")
                  )
                )
              ),

              if (length(tool_names) > 0) {
                div(
                  class = "agent-tool-log",
                  icon("wrench"),
                  strong(" Tools used: "),
                  paste(
                    tool_names,
                    collapse = ", "
                  )
                )
              },

              if (isTRUE(entry$success)) {
                div(
                  class = "agent-model",
                  paste("Model:", entry$model)
                )
              }
            )
          )
        })
      )
    })
  })
}
