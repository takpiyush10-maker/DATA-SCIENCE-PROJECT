from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BUILDINGS = {'Academic Block': (12, 100), 'Library': (9, 65), 'Hostel': (18, 140), 'Laboratory': (16, 55), 'Administration': (7, 45)}

def generate_data(days=120, seed=42):
    rng = np.random.default_rng(seed)
    times = pd.date_range('2025-01-01', periods=24*days, freq='h')
    frames = []
    for building, (base, capacity) in BUILDINGS.items():
        hour = times.hour.to_numpy()
        active = ((hour >= 8) & (hour < 18) & (times.dayofweek.to_numpy() < 5))
        if building == 'Hostel':
            active = (hour < 8) | (hour >= 18)
        if building == 'Library':
            active = (hour >= 8) & (hour < 22)
        occ = np.maximum(0, rng.normal(np.where(active, capacity*.7, capacity*.06), capacity*.06)).astype(int)
        temp = 27 + 7*np.sin(2*np.pi*(hour-9)/24) + rng.normal(0, 1.5, len(times))
        appliances = np.maximum(1, (occ*.3 + rng.normal(4, 2, len(times))).astype(int))
        usage = base + occ*.18 + appliances*.35 + np.maximum(temp-25, 0)*1.1 + rng.normal(0, 1.7, len(times))
        injected = rng.random(len(times)) < .015
        usage[injected] += rng.uniform(20, 65, injected.sum())
        usage = np.maximum(usage, .1)
        frame = pd.DataFrame({'timestamp': times, 'building': building, 'energy_kwh': usage.round(3), 'occupancy': occ, 'appliances_on': appliances, 'temperature_c': temp.round(2), 'is_holiday': (times.dayofweek >= 5).astype(int), 'injected_anomaly': injected.astype(int)})
        missing = rng.choice(len(frame), size=max(1,len(frame)//200), replace=False)
        frame.loc[missing, 'energy_kwh'] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def clean_data(raw):
    required = {'timestamp', 'building', 'energy_kwh'}
    if not required.issubset(raw.columns):
        raise ValueError('Required columns: timestamp, building, energy_kwh')
    df = raw.copy()
    df['timestamp'] = pd.to_datetime(df.timestamp, errors='coerce')
    if getattr(df.timestamp.dt, 'tz', None) is not None:
        df['timestamp'] = df.timestamp.dt.tz_localize(None)
    df['building'] = df.building.astype('string').str.strip().replace('', pd.NA)
    bad_keys = df.timestamp.isna() | df.building.isna()
    report = {'input_rows':len(df), 'invalid_keys_removed':int(bad_keys.sum())}
    df = df.loc[~bad_keys].copy()
    if df.empty:
        raise ValueError('No valid timestamp/building rows.')
    if (df.timestamp != df.timestamp.dt.floor('h')).any():
        raise ValueError('Use exactly hourly interval data with timestamps on the hour; aggregate sub-hourly interval kWh first.')
    report['duplicate_rows_removed'] = int(df.duplicated(['building','timestamp']).sum())
    df = df.drop_duplicates(['building','timestamp'], keep='last')
    df['energy_kwh'] = pd.to_numeric(df.energy_kwh, errors='coerce')
    bad = ~np.isfinite(df.energy_kwh) | (df.energy_kwh < 0)
    report['invalid_or_missing_energy'] = int(bad.sum())
    df.loc[bad, 'energy_kwh'] = np.nan
    frames = []
    added = 0
    for b,g in df.groupby('building'):
        idx = pd.date_range(g.timestamp.min(), g.timestamp.max(), freq='h')
        if len(idx) > 100000:
            raise ValueError('Building history too long; limit upload to 100,000 hourly intervals per building.')
        added += len(idx)-len(g)
        g = g.set_index('timestamp').reindex(idx).rename_axis('timestamp')
        g['building'] = b
        g['energy_observed'] = g.energy_kwh.notna()
        # Past-only fill; never use a future meter observation.
        g['energy_kwh'] = g.energy_kwh.ffill(limit=3)
        g['is_holiday'] = pd.to_numeric(g.get('is_holiday', pd.Series(0,index=g.index)),errors='coerce').fillna(0).clip(0,1).astype(int)
        frames.append(g.reset_index())
    report['missing_hour_rows_added'] = added
    result = pd.concat(frames, ignore_index=True).sort_values(['building','timestamp']).reset_index(drop=True)
    report['output_rows'] = len(result)
    report['unfilled_energy_rows'] = int(result.energy_kwh.isna().sum())
    return result, report

FEATURES = ['hour_sin','hour_cos','weekday','month','is_weekend','is_holiday','lag_1','lag_24','lag_168','rolling_24']

def make_features(clean):
    df = clean.sort_values(['building','timestamp']).copy()
    df['hour_sin'] = np.sin(2*np.pi*df.timestamp.dt.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*df.timestamp.dt.hour/24)
    df['weekday'] = df.timestamp.dt.dayofweek
    df['month'] = df.timestamp.dt.month
    df['is_weekend'] = (df.weekday >= 5).astype(int)
    groups = df.groupby('building').energy_kwh
    for lag in [1,24,168]:
        df[f'lag_{lag}'] = groups.shift(lag)
    df['rolling_24'] = groups.transform(lambda s: s.shift(1).rolling(24,min_periods=24).mean())
    return df

def matrix(df, columns=None):
    x = pd.get_dummies(df[FEATURES+['building']], columns=['building'], dtype=float)
    return x if columns is None else x.reindex(columns=columns, fill_value=0)

def train_evaluate(clean):
    f = make_features(clean)
    f = f.loc[f.energy_observed].dropna(subset=FEATURES+['energy_kwh'])
    dates = np.sort(f.timestamp.unique())
    if len(dates) < 72:
        raise ValueError('Need at least 10 complete days per building; 30+ days recommended.')
    cutoff = pd.Timestamp(dates[int(len(dates)*.8)])
    train, test = f[f.timestamp < cutoff].copy(), f[f.timestamp >= cutoff].copy()
    if set(test.building)-set(train.building):
        raise ValueError('A test building lacks training history.')
    x = matrix(train)
    model = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=3, random_state=42,n_jobs=-1)
    model.fit(x,train.energy_kwh)
    test['predicted_kwh'] = model.predict(matrix(test,x.columns))
    test['baseline_kwh'] = test.lag_24
    def metrics(y,p):
        return {'MAE_kWh':float(mean_absolute_error(y,p)), 'RMSE_kWh':float(np.sqrt(mean_squared_error(y,p))), 'R2':float(r2_score(y,p))}
    scores = {'random_forest':metrics(test.energy_kwh,test.predicted_kwh),'previous_day_baseline':metrics(test.energy_kwh,test.baseline_kwh), 'train_rows':len(train),'test_rows':len(test),'test_start':str(cutoff),'evaluation':'Chronological holdout, rolling one-hour-ahead predictions with actual earlier test observations available.'}
    detector = IsolationForest(n_estimators=100, contamination=.02, random_state=42,n_jobs=-1)
    anomaly_cols = ['energy_kwh','lag_24','rolling_24','hour_sin','hour_cos','is_weekend']
    detector.fit(train[anomaly_cols])
    test['anomaly'] = detector.predict(test[anomaly_cols]) == -1
    test['anomaly_score'] = -detector.decision_function(test[anomaly_cols])
    test['positive_excess_kwh'] = np.maximum(test.energy_kwh-test.predicted_kwh,0)
    test['review_candidate'] = test.anomaly & (test.positive_excess_kwh > 0)
    # Refit a production forecasting model after reporting untouched holdout scores.
    all_x = matrix(f)
    production = RandomForestRegressor(n_estimators=100,max_depth=16,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(all_x,f.energy_kwh)
    importance = pd.DataFrame({'feature':x.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
    return {'model':production,'columns':all_x.columns,'scores':scores,'test':test,'importance':importance}

def forecast(clean, fitted, hours=24, holidays=()):
    rows = []
    holiday_set = set(holidays)
    for b,g in clean.groupby('building'):
        hist = g.sort_values('timestamp').copy()
        if len(hist)<168 or hist.energy_kwh.tail(168).isna().any():
            continue
        values = hist.energy_kwh.tolist()
        last = hist.timestamp.iloc[-1]
        for step in range(1,hours+1):
            ts = last+pd.Timedelta(hours=step)
            row = {'timestamp':ts,'building':b,'hour_sin':np.sin(2*np.pi*ts.hour/24),'hour_cos':np.cos(2*np.pi*ts.hour/24),'weekday':ts.dayofweek,'month':ts.month,'is_weekend':int(ts.dayofweek>=5),'is_holiday':int(str(ts.date()) in holiday_set),'lag_1':values[-1],'lag_24':values[-24],'lag_168':values[-168],'rolling_24':float(np.mean(values[-24:]))}
            pred = max(0,float(fitted['model'].predict(matrix(pd.DataFrame([row]),fitted['columns']))[0]))
            values.append(pred)
            rows.append({'timestamp':ts,'building':b,'forecast_kwh':pred})
    return pd.DataFrame(rows,columns=['timestamp','building','forecast_kwh'])

def recommendations(test, tariff=8, recoverable_fraction=.25):
    rows = []
    for b,g in test.groupby('building'):
        candidates = g[g.review_candidate]
        excess = float(candidates.positive_excess_kwh.sum())
        rows.append({'building':b,'review_alerts':len(candidates),'flagged_positive_excess_kwh':round(excess,2),'scenario_recoverable_kwh':round(excess*recoverable_fraction,2),'scenario_saving_inr':round(excess*recoverable_fraction*tariff,2),'action':'Review alert timestamps, occupancy, HVAC schedules and equipment; confirm waste before taking action.'})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--output',default='results')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    raw = pd.read_csv(args.csv) if args.csv else generate_data()
    raw.to_csv(out/'input_data.csv',index=False)
    clean, quality = clean_data(raw)
    fitted = train_evaluate(clean)
    clean.to_csv(out/'cleaned_data.csv',index=False)
    fitted['test'].to_csv(out/'holdout_predictions_alerts.csv',index=False)
    forecast(clean,fitted).to_csv(out/'next_24_hours.csv',index=False)
    recommendations(fitted['test']).to_csv(out/'recommendations.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(fitted['scores'],indent=2))
    (out/'quality.json').write_text(json.dumps(quality,indent=2))
    print(json.dumps(fitted['scores'],indent=2))
