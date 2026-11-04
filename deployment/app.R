
library(shiny)
library(bslib)
library(plotly)
library(DT)
library(dplyr)
library(readr)
library(jsonlite)
library(scales)
library(htmltools)

dashboard_env <- environment()

source("helpers.R", local = dashboard_env)
source("modules/home.R", local = dashboard_env)
source("modules/overview.R", local = dashboard_env)
source("modules/posterior.R", local = dashboard_env)
source("modules/prediction.R", local = dashboard_env)
source("modules/model_assessment.R", local = dashboard_env)
source("modules/ai_assistant.R", local = dashboard_env)

source("dashboard_ui.R", local = dashboard_env)
source("dashboard_server.R", local = dashboard_env)

shinyApp(
  ui = ui,
  server = server
)