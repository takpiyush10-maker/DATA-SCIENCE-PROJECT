import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from energy import generate_data, clean_data, train_evaluate, forecast, recommendations

st.set_page_config(page_title='Smart Campus Energy Analyzer',layout='wide')
st.title('Smart Campus Energy Consumption Tracker & Analyzer')
st.caption('Forecast demand • Investigate unusual consumption • Evaluate saving scenarios')
st.sidebar.header('Data and assumptions')
upload = st.sidebar.file_uploader('Upload hourly interval readings (CSV)',type=['csv'])
tariff = st.sidebar.number_input('Energy tariff (INR/kWh): scenario input',min_value=0.,value=8.,step=.5)
recovery = st.sidebar.slider('Assumed recoverable fraction of flagged excess',0,100,25)/100
horizon = st.sidebar.slider('Forecast horizon (hours)',1,48,24)
holidays = st.sidebar.text_input('Future holiday dates, comma-separated',help='YYYY-MM-DD; blank means no special holidays.')
st.sidebar.info('energy_kwh means energy consumed during ONE hour, not a cumulative meter register. Demo data are synthetic; costs exclude fixed charges and taxes.')

@st.cache_data
def load(payload):
    import io
    return clean_data(pd.read_csv(io.BytesIO(payload)) if payload else generate_data())

@st.cache_resource
def fit_cached(clean):
    return train_evaluate(clean)

try:
    clean, quality = load(upload.getvalue() if upload else None)
    fitted = fit_cached(clean)
except Exception as exc:
    st.error(str(exc)); st.stop()
if not upload:
    st.warning('SYNTHETIC DEMONSTRATION: not actual campus consumption or verified savings.')
selected = st.sidebar.multiselect('Buildings',sorted(clean.building.unique()),default=sorted(clean.building.unique()))
if not selected:
    st.info('Select at least one building.'); st.stop()
df = clean[clean.building.isin(selected)].copy()
observed = df[df.energy_observed]
test = fitted['test'][fitted['test'].building.isin(selected)]
a,b,c,d = st.columns(4)
a.metric('Observed energy (kWh)',f'{observed.energy_kwh.sum():,.1f}')
b.metric('Observed energy cost scenario',f'INR {observed.energy_kwh.sum()*tariff:,.0f}')
c.metric('Holdout unusual readings',int(test.anomaly.sum()))
d.metric('Missing original readings',int((~df.energy_observed).sum()))
tabs = st.tabs(['1–4 Data & cleaning','5 Exploratory analysis','6–8 Models & alerts','9 Forecast & recommendations'])
with tabs[0]:
    st.subheader('Problem: identify peak consumption and readings requiring investigation')
    st.write('Data generation/upload → hourly cleaning → EDA → lag features → chronological model evaluation → forecast → dashboard.')
    st.json(quality)
    st.dataframe(df.head(200),use_container_width=True)
    st.download_button('Download cleaned data',df.to_csv(index=False),'cleaned_data.csv','text/csv')
    st.write('Missing energy is forward-filled for at most 3 hours for feature continuity; imputed readings are never training targets. Original observed totals exclude imputation.')
with tabs[1]:
    st.subheader('Campus daily energy (observed readings only)')
    st.line_chart(observed.assign(day=observed.timestamp.dt.floor('d')).groupby('day').energy_kwh.sum())
    st.subheader('Building comparison')
    st.bar_chart(observed.groupby('building').energy_kwh.sum())
    st.subheader('Average hourly pattern')
    st.line_chart(observed.assign(hour=observed.timestamp.dt.hour).groupby(['hour','building']).energy_kwh.mean().unstack())
    st.subheader('Weekday vs weekend average hourly reading')
    st.bar_chart(observed.assign(day_type=observed.timestamp.dt.dayofweek.map(lambda x:'Weekend' if x>=5 else 'Weekday')).groupby('day_type').energy_kwh.mean())
    if 'temperature_c' in observed.columns:
        fig,ax=plt.subplots(figsize=(8,3)); ax.scatter(pd.to_numeric(observed.temperature_c,errors='coerce'),observed.energy_kwh,s=4,alpha=.25)
        ax.set(xlabel='Temperature (C)',ylabel='Hourly energy (kWh)'); st.pyplot(fig); plt.close(fig)
    st.caption('Compare data completeness before interpreting totals. Summed hourly interval kWh numerically equals average campus kW over that hour, not an instantaneous demand peak.')
with tabs[2]:
    st.subheader('Chronological 80/20 holdout evaluation')
    st.json(fitted['scores'])
    st.info('Scores cover all buildings. Holdout predictions are one-hour-ahead with earlier actual readings available; future forecasts are recursive and have different error behavior.')
    st.write('Predictors: building, cyclical hour, weekday, month, weekend, holiday, 1/24/168-hour lags and a past-only 24-hour rolling mean. Occupancy and temperature are exploratory only: future values are not assumed known.')
    building = st.selectbox('Prediction plot building',selected)
    plot = test[test.building==building].tail(240)
    fig,ax=plt.subplots(figsize=(11,4)); ax.plot(plot.timestamp,plot.energy_kwh,label='Actual',linewidth=1); ax.plot(plot.timestamp,plot.predicted_kwh,label='Predicted',linewidth=1)
    unusual=plot[plot.anomaly]; ax.scatter(unusual.timestamp,unusual.energy_kwh,color='red',s=22,label='Isolation Forest alert'); ax.legend(); ax.set_ylabel('kWh per hour'); fig.autofmt_xdate(); st.pyplot(fig); plt.close(fig)
    st.dataframe(test[test.anomaly][['timestamp','building','energy_kwh','predicted_kwh','anomaly_score','review_candidate']],use_container_width=True)
    st.bar_chart(fitted['importance'].head(12).set_index('feature'))
    st.caption('An anomaly is not proof of waste. Isolation Forest is fitted only on training data with a 2% contamination assumption; unusual low usage can also trigger alerts.')
    st.download_button('Download holdout predictions and alerts',test.to_csv(index=False),'predictions_alerts.csv','text/csv')
with tabs[3]:
    future = forecast(clean,fitted,horizon,[s.strip() for s in holidays.split(',') if s.strip()])
    future = future[future.building.isin(selected)]
    st.subheader('Future demand: recursive hourly forecast')
    if future.empty:
        st.warning('Forecast requires a complete last 168-hour energy history after cleaning.')
    else:
        st.line_chart(future.pivot(index='timestamp',columns='building',values='forecast_kwh'))
        st.dataframe(future,use_container_width=True)
        st.download_button('Download forecast',future.to_csv(index=False),'forecast.csv','text/csv')
    rec = recommendations(test,tariff,recovery)
    st.subheader('Investigation priorities and conditional savings')
    st.dataframe(rec,use_container_width=True)
    st.warning('Savings are an illustrative scenario over the HOLDOUT period, not verified savings or an annual estimate. Flagged excess = max(actual − model prediction, 0) for high-side anomaly alerts. Scenario savings = flagged excess × assumed recoverable fraction × tariff.')
    st.write('Inspect unoccupied-hour HVAC and lighting; check lab equipment schedules; verify meter calibration; compare occupancy before changing operations. Do not automatically switch off safety-critical equipment.')
    st.download_button('Download recommendations',rec.to_csv(index=False),'recommendations.csv','text/csv')
