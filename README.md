# Customer Purchase Behavior Prediction - Setup on a new laptop

## Folder layout (keep exactly like this)
```
customer_purchase_project/
  app.py                 <- Streamlit app (run from THIS folder)
  requirements.txt
  run_app.bat            <- Windows: double-click
  run_app.sh             <- Mac/Linux
  models/
    best_model.pkl       <- final XGBoost model
    scaler.pkl           <- StandardScaler (23 features)
  notebooks/EDA.ipynb    <- full notebook with saved outputs
  images/                <- optional: put banner.png here (app skips it if missing)
```
app.py loads "models/best_model.pkl" with a relative path, so it must be launched from the project folder.

## Run
1. Install Python 3.11 or 3.12 (tick "Add Python to PATH" on Windows).
2. Windows: double-click run_app.bat. Mac/Linux: ./run_app.sh
   (first run creates .venv and installs packages; needs internet once)
3. Browser opens at http://localhost:8501

Manual: `pip install -r requirements.txt` then `streamlit run app.py`

## Why these versions
- scaler.pkl was saved with scikit-learn 1.9.0, so it is pinned.
- app.py uses `use_container_width`, which newer Streamlit versions deprecate, so Streamlit is pinned to 1.45.1.

## Demo inputs (defaults in the app)
Age 30, Income 50000, Purchases 10, Tenure 5, Last purchase 30 days, Time on site 25, Sessions 12.

## Not available (lost with the old laptop)
customerData_500k.csv (dataset) - only needed to re-run the notebook, NOT the app.
