# Student Service Request Monitoring - SLA Breach Prediction (Streamlit)

Deployment of the **Section B midterm** (Decision Tree, 40 service requests) as a Streamlit
decision-support app for student services staff. Predictions are advisory; a human records the final decision.

## Pages
Home | Dataset (upload + validation) | Data Analysis (EDA) | Prediction (saved model) | Dashboard (KPIs, review queue, decision log) | System Design (requirements, use case diagram, workflow)

## Repository layout
```
app.py                      Streamlit entry point
train_model.py              Midterm training code -> saves models/trained_model.joblib (run once)
requirements.txt
modules/                    data_processing, exploratory_analysis, prediction, dashboard, system_design
models/trained_model.joblib Fitted preprocessing + DecisionTree pipeline
models/model_metadata.json  Metrics, features, version, midterm test cases
data/cleaned_student_requests.csv   Synthetic midterm dataset (no real student records)
assets/                     use_case_diagram.png, decision_workflow.png
notebooks/                  Original midterm notebook (miles_midterm_original.ipynb)
tests/test_logic.py         pytest checks (model matches midterm, validation, cleaning)
```

## Run locally
```
pip install -r requirements.txt
python train_model.py        # optional: re-creates the .joblib in YOUR environment
streamlit run app.py
pytest -q
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. https://share.streamlit.io -> sign in -> **Create app** -> pick the repo and branch.
3. Main file path: `app.py`. Advanced settings -> Python 3.12. Deploy; open **Manage app** for logs if the build fails.

If the model file cannot be read (scikit-learn version mismatch) the app rebuilds it once from the same midterm code and shows a warning.

## Model facts (from the saved metadata)
Features: Service_Type, Priority, Queue_Category, Assigned_Office, Submission_Weekday (no leakage fields).
Midterm hold-out (8 cases): accuracy 0.875, precision 0.80, recall 1.00, F1 0.889, baseline 0.50.

## Limitations
40 synthetic rows; 8 test cases; probabilities are tree-leaf proportions, not calibrated; roles are a demo selector, not authentication; the decision log lives in session memory only.
