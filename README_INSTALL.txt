Bayesian Credit Risk Intelligence — Dashboard Update

This archive contains complete replacements for dashboard/app.R, dashboard/ui.R,
dashboard/server.R, dashboard/www/custom.css, all six modules, and a new dashboard/helpers.R.

1. Make a backup of your current dashboard directory.
2. Extract the archive into your existing Bayesian-Credit-Risk-Intelligence project root.
3. Do not remove data/processed or results/tables.
4. Install dependencies if necessary:
   R -e 'install.packages(c("shiny","bslib","plotly","DT","dplyr","readr","jsonlite","scales","htmltools"), repos="https://cloud.r-project.org")'
5. From project root, run:
   cd dashboard
   R -e 'shiny::runApp(".", host="127.0.0.1", port=3838)'
6. Open http://127.0.0.1:3838/

Notes:
- Credit limits in the historical Taiwan dataset are denominated in NT$, not US dollars.
- The AI assistant remains deterministic/rule-based; this update does not claim LLM integration.
- On the Home page, LinkedIn is the URL provided by the user.
- Data files and analysis are read from the existing project, not included in this archive.
