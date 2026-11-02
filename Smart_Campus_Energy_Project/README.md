# Smart Campus Energy Consumption Tracker & Analyzer
Author: PIYUSH TAK

## Start here
Use Python 3.10 or newer. Extract the ZIP, open a terminal inside the project folder, and run:

```bash
python -m venv .venv
```
Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open the local URL printed by Streamlit. Internet is needed for initial dependency installation, not for the synthetic demo after installation.

## Command-line analysis
```bash
python energy.py --output results
python energy.py --csv data/synthetic_campus_energy.csv --output my_results
python -m unittest test_energy.py
```
The dashboard also accepts your own CSV. Streamlit is not required for command-line processing.

## Nine-stage workflow
1. Problem definition: identify high consumption, unusual readings and investigation priorities.
2. Dataset creation: reproducible synthetic data, 120 days, five buildings, 14,400 hourly rows.
3. Data collection: CSV template includes timestamp, building, hourly energy_kwh, occupancy, appliances_on, temperature_c, is_holiday.
4. Cleaning: parse times, remove invalid keys, deduplicate (keep last), invalidate negative/nonfinite energy, restore missing hours, forward-fill at most 3 hours. No backfilling. Keep observed/imputed flag.
5. EDA: hourly/daily patterns, building comparison, weekday/weekend comparison, temperature relationship.
6. Features: cyclical hour, weekday, month, weekend, holiday, lag 1/24/168, past-only rolling 24-hour mean, building indicators.
7. Modeling: Random Forest regression; previous-day baseline; Isolation Forest anomaly detector fitted on training only.
8. Prediction visualization: actual/predicted chart, anomaly markers, metrics and feature importances.
9. Dashboard: uploads, filters, quality report, demand forecasts, investigation alerts, tariff scenarios and CSV downloads.

## Data contract
Required: timestamp, building, energy_kwh. Extra columns are optional.
Timestamps must be local campus time with one reading exactly on each hour, per building.
energy_kwh MUST be consumption during the hour. It is NOT a cumulative meter reading, watts or instantaneous kW. Convert cumulative readings to interval differences separately, treating resets and rollovers explicitly.
For sub-hourly readings, sum interval energy by building/hour before upload; do not sum cumulative registers.
Use at least 10 complete days per building, preferably 30+ days, and seven complete trailing days for future forecasting.
Holiday flags: 0 or 1. Synthetic flags use weekends as a stand-in; replace with an actual campus calendar. Forecast special holidays are entered separately in the sidebar.
No tariff, weather, holiday or campus records are fetched automatically. INR 8/kWh is a user-editable demo assumption, not a verified local tariff.
Synthetic injected_anomaly is for demonstration only; it is excluded from predictive inputs.

## Evaluation and limitations
The common time cutoff holds out the final 20% of usable timestamps, across buildings. Earlier actual holdout readings are available for later one-hour-ahead predictions: this is rolling one-step evaluation, NOT a frozen 24-hour forecast evaluation.
A separate production forest is refitted on all usable observed historical targets for recursive 1-48 hour forecasting. Future steps reuse predictions; uncertainty grows. No calibrated prediction intervals are supplied. Do not interpret one-step holdout metrics as multi-step forecast metrics.
Occupancy, appliances and temperature appear in EDA but are not forecast predictors because their future values are unknown.
Missing targets are excluded from fitting/evaluation. Past-only fills may support lag features; long gaps cause feature rows to be excluded. Building totals use only observed energy, so gaps can understate totals. No area/occupancy normalization is claimed.
Isolation Forest contamination=0.02 is a modeling assumption, not a known true waste rate. Alerts can be legitimate and include unusually low demand. Confirm alerts manually.
Illustrative savings = sum(max(actual-predicted,0) for high-side anomaly flags) * assumed recoverable fraction * tariff. Values cover only the holdout period; they are not verified savings, not guaranteed recoverable energy, and not annualized. Fixed/demand charges and taxes are excluded.
This is a local batch analytics prototype, not live IoT ingestion or automated equipment control.

## Included deliverables
app.py, energy.py, requirements.txt, test_energy.py, analysis.ipynb, project report, synthetic dataset, sample outputs, metrics, recommendations and chart.
Core data/model workflow was executed on the included synthetic data. The interactive browser dashboard has not been manually browser-tested here.

## Technical documentation
- https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.IsolationForest.html
- https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- https://docs.streamlit.io/1.25.0/develop/api-reference/widgets/st.file_uploader
