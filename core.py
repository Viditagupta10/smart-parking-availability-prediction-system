"""
core.py - Smart Parking Availability & Prediction System
All data-science logic lives here (no Streamlit code), so it is easy to test.

Two prediction models:
  1. SHORT-TERM  (next 1/2/3 hours)  - uses live occupancy + lags
  2. PLANNING    (any future date)   - uses calendar + weather + events only
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATUS_LABELS = ["Available", "Filling up", "Almost full"]
STATUS_BINS = [0.70, 0.90]          # <70% available, 70-90% filling up, >=90% almost full
HORIZONS = (1, 2, 3)


def status_of(rate):
    return STATUS_LABELS[int(np.digitize(rate, STATUS_BINS))]


# --------------------------------------------------------------------------
# 1. DATA  (synthetic demo / CSV loader / SQLite)
# --------------------------------------------------------------------------
DEMO_LOTS = [
    # name, type, capacity, lat, lon
    ("Tech Park Office", "Office", 300, 28.6139, 77.2090),
    ("City Mall", "Mall", 500, 28.6200, 77.2200),
    ("Central Hospital", "Hospital", 200, 28.6080, 77.2150),
    ("Railway Station", "Station", 250, 28.6280, 77.2060),
    ("Old Market", "Market", 150, 28.6170, 77.2300),
]


def _g(h, mu, sd):
    return np.exp(-((h - mu) / sd) ** 2)


def _base_profile(kind, h, off_day):
    """Typical occupancy rate (0-1) by hour for each type of parking lot."""
    if kind == "Office":
        return np.where(off_day, 0.10 + 0.10 * _g(h, 13, 4), 0.10 + 0.82 * _g(h, 12, 3.5))
    if kind == "Mall":
        return np.where(off_day, 0.25 + 0.65 * _g(h, 16, 4), 0.15 + 0.55 * _g(h, 18, 3))
    if kind == "Hospital":
        return (0.35 + 0.45 * _g(h, 11, 4.5)) * np.where(off_day, 0.9, 1.0)
    if kind == "Station":
        return np.where(off_day, 0.20 + 0.30 * _g(h, 12, 4),
                        0.20 + 0.50 * _g(h, 8.5, 1.8) + 0.45 * _g(h, 18.5, 2))
    return 0.10 + (0.60 + 0.15 * off_day) * _g(h, 19, 2.5)   # Market


def generate_demo_data(days=150, seed=7, end=None):
    """Hourly occupancy for 5 lots with weekday/weekend, holidays, rain and events."""
    rng = np.random.default_rng(seed)
    # demo "now" = yesterday 5 PM (busy time, so the live view is interesting)
    end = pd.Timestamp(end) if end is not None else (pd.Timestamp.today().floor("D") - pd.Timedelta(hours=7))
    ts = pd.date_range(end=end, periods=days * 24, freq="h")
    day = ts.floor("D")
    udays = day.unique()
    holidays = set(rng.choice(udays[: len(udays) - 1], size=max(3, days // 30), replace=False))
    rainy = {d: rng.integers(0, 24 - 3) for d in udays if rng.random() < 0.2}   # rain start hour

    is_hol = np.array([d in holidays for d in day], dtype=int)
    off = ((ts.dayofweek >= 5) | (is_hol == 1)).astype(float)
    hour = ts.hour.values
    rain = np.zeros(len(ts))
    for i, (d, hh) in enumerate(zip(day, hour)):
        st_h = rainy.get(d)
        if st_h is not None and st_h <= hh < st_h + 3:
            rain[i] = rng.uniform(2, 8)

    frames, meta = [], []
    for name, kind, cap, lat, lon in DEMO_LOTS:
        rate = _base_profile(kind, hour.astype(float), off)
        rate = rate + np.where(rain > 0, -0.12 if kind == "Market" else 0.08, 0.0)
        ev = np.zeros(len(ts), dtype=int)
        if kind in ("Mall", "Market", "Station"):       # evening events (sale / festival / big train)
            for d in udays:
                if rng.random() < 0.07:
                    ev[(day == d) & (hour >= 17) & (hour < 22)] = 1
        rate = rate + 0.25 * ev
        noise = np.zeros(len(ts))
        for i in range(1, len(ts)):                      # AR(1) noise -> smooth, realistic
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.03)
        rate = np.clip(rate + noise, 0.02, 1.0)
        frames.append(pd.DataFrame({
            "ts": ts, "lot_id": name, "occupied": np.round(rate * cap).astype(int), "capacity": cap,
            "is_holiday": is_hol, "rain": rain.round(1), "event": ev}))
        meta.append({"lot_id": name, "type": kind, "capacity": cap, "lat": lat, "lon": lon})
    df = pd.concat(frames, ignore_index=True)
    df["rate"] = (df["occupied"] / df["capacity"]).clip(0, 1)
    return df, pd.DataFrame(meta)


def load_csv(file):
    """
    CSV columns (names flexible):
      timestamp, lot_id, occupied      (required)
      capacity, rain, is_holiday, event, lat, lon   (optional)
    Any time-resolution is converted to hourly average per lot.
    """
    raw = pd.read_csv(file, sep=None, engine="python")
    raw.columns = [c.strip().lower() for c in raw.columns]

    def pick(*names):
        return next((c for c in raw.columns if c in names), None)

    c_ts = pick("timestamp", "datetime", "time", "date")
    c_lot = pick("lot_id", "lot", "parking", "parking_lot", "location", "name", "systemcodenumber")
    c_occ = pick("occupied", "occupancy", "vehicles", "count", "cars")
    if not (c_ts and c_lot and c_occ):
        raise ValueError("CSV me 'timestamp', 'lot_id' aur 'occupied' columns hone chahiye.")
    c_cap, c_rain = pick("capacity", "total", "spaces"), pick("rain", "rain_mm", "precipitation")
    c_hol, c_ev = pick("is_holiday", "holiday"), pick("event")

    ts = pd.to_datetime(raw[c_ts], errors="coerce")
    if ts.isna().mean() > 0.2:
        ts = pd.to_datetime(raw[c_ts], errors="coerce", dayfirst=True)
    d = pd.DataFrame({"ts": ts, "lot_id": raw[c_lot].astype(str),
                      "occupied": pd.to_numeric(raw[c_occ], errors="coerce")})
    d["capacity"] = pd.to_numeric(raw[c_cap], errors="coerce") if c_cap else np.nan
    d["rain"] = pd.to_numeric(raw[c_rain], errors="coerce") if c_rain else 0.0
    d["is_holiday"] = pd.to_numeric(raw[c_hol], errors="coerce") if c_hol else 0.0
    d["event"] = pd.to_numeric(raw[c_ev], errors="coerce") if c_ev else 0.0
    d = d.dropna(subset=["ts", "occupied"])

    out, meta = [], []
    for lot, g in d.groupby("lot_id"):
        g = g.set_index("ts").sort_index()
        h = g.resample("h").agg({"occupied": "mean", "capacity": "max", "rain": "mean",
                                 "is_holiday": "max", "event": "max"})
        h["occupied"] = h["occupied"].interpolate(limit_direction="both")
        cap = h["capacity"].max()
        cap = cap if pd.notna(cap) else float(np.ceil(h["occupied"].max() * 1.1))
        h["capacity"] = cap
        for c in ("rain", "is_holiday", "event"):
            h[c] = h[c].fillna(0)
        if len(h) < 24 * 14:
            raise ValueError(f"'{lot}' ka kam se kam 14 din ka data chahiye.")
        h = h.reset_index()
        h["lot_id"] = lot
        out.append(h)
        row = raw[raw[c_lot].astype(str) == lot]
        lat = pd.to_numeric(row[pick("lat", "latitude")], errors="coerce").mean() if pick("lat", "latitude") else np.nan
        lon = pd.to_numeric(row[pick("lon", "lng", "longitude")], errors="coerce").mean() if pick("lon", "lng", "longitude") else np.nan
        meta.append({"lot_id": lot, "type": "Custom", "capacity": cap, "lat": lat, "lon": lon})
    df = pd.concat(out, ignore_index=True)
    df["occupied"] = df["occupied"].round().astype(int)
    df["rate"] = (df["occupied"] / df["capacity"]).clip(0, 1)
    return df, pd.DataFrame(meta)


def save_to_sqlite(df, db_path="parking.db", table="parking_history"):
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)


def load_from_sqlite(db_path="parking.db", table="parking_history"):
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(f"SELECT * FROM {table}", con, parse_dates=["ts"])


# --------------------------------------------------------------------------
# 2. FEATURES
# --------------------------------------------------------------------------
def _calendar(ts):
    ts = pd.DatetimeIndex(ts)
    return pd.DataFrame({
        "hour": ts.hour, "dow": ts.dayofweek, "is_weekend": (ts.dayofweek >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
    })


def _lot_onehot(lots, all_lots):
    return pd.DataFrame({f"lot_{l}": (np.asarray(lots) == l).astype(int) for l in all_lots})


def build_short(df, all_lots, h):
    """Features known at time t  ->  target = occupancy rate at t+h hours."""
    d = df.sort_values(["lot_id", "ts"]).reset_index(drop=True)
    g = d.groupby("lot_id")["rate"]
    d["rate_now"], d["rate_l1"], d["rate_l2"], d["rate_l24"] = d["rate"], g.shift(1), g.shift(2), g.shift(24)
    d["roll3"] = g.transform(lambda s: s.rolling(3).mean())
    d["target"] = g.shift(-h)
    d["t_ts"] = d["ts"] + pd.Timedelta(hours=h)
    for c in ("is_holiday", "rain", "event"):                 # conditions at the TARGET time
        d["t_" + c] = d.groupby("lot_id")[c].shift(-h)
    d = d.dropna(subset=["rate_l1", "rate_l2", "rate_l24", "roll3", "target", "t_rain"]).reset_index(drop=True)
    X = pd.concat([
        d[["rate_now", "rate_l1", "rate_l2", "rate_l24", "roll3"]],
        _calendar(d["t_ts"]),
        d[["t_is_holiday", "t_rain", "t_event"]].rename(columns=lambda c: c[2:]),
        _lot_onehot(d["lot_id"], all_lots)], axis=1)
    return X, d["target"], d["ts"], d["rate_now"]


def build_plan(df, all_lots):
    """Calendar/weather features only (no lags) -> can predict ANY future date/time."""
    d = df.sort_values(["lot_id", "ts"]).reset_index(drop=True)
    X = pd.concat([_calendar(d["ts"]), d[["is_holiday", "rain", "event"]].reset_index(drop=True),
                   _lot_onehot(d["lot_id"], all_lots)], axis=1)
    return X, d["rate"], d["ts"], d


# --------------------------------------------------------------------------
# 3. MODELS + EVALUATION (chronological split -> no data leakage)
# --------------------------------------------------------------------------
def _models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, random_state=42),
    }


def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.clip(np.asarray(y_pred), 0, 1)
    err = y_true - y_pred
    st_true, st_pred = np.digitize(y_true, STATUS_BINS), np.digitize(y_pred, STATUS_BINS)
    return {"MAE (pts %)": round(np.mean(np.abs(err)) * 100, 2),
            "RMSE (pts %)": round(np.sqrt(np.mean(err ** 2)) * 100, 2),
            "R2": round(r2_score(y_true, y_pred), 3),
            "Within 10 pts %": round(np.mean(np.abs(err) <= 0.10) * 100, 1),
            "Status accuracy %": round(np.mean(st_true == st_pred) * 100, 1)}


def _time_split(ts, frac=0.2):
    cut = np.sort(ts.unique())[int(ts.nunique() * (1 - frac))]
    train = (ts < cut).values
    return train, ~train


def _importance(model, Xte, yte):
    n = min(2500, len(Xte))
    idx = np.random.default_rng(0).choice(len(Xte), n, replace=False)
    r = permutation_importance(model, Xte.iloc[idx], yte.iloc[idx], n_repeats=3, random_state=0, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=Xte.columns)
    lot_cols = [c for c in imp.index if c.startswith("lot_")]
    imp = pd.concat([imp.drop(lot_cols), pd.Series({"lot (which parking)": imp[lot_cols].sum()})])
    imp = imp.clip(lower=0)
    return (imp / imp.sum()).sort_values(ascending=False)


def train_short(df, all_lots):
    # model comparison at horizon = 1 hour
    X, y, ts, rate_now = build_short(df, all_lots, 1)
    tr, te = _time_split(ts)
    rows = [{"Model": "Baseline (persistence: same as now)", **evaluate(y[te], rate_now[te])}]
    for name, m in _models().items():
        m.fit(X[tr], y[tr])
        rows.append({"Model": name, **evaluate(y[te], m.predict(X[te]))})
    comp = pd.DataFrame(rows).sort_values("MAE (pts %)").reset_index(drop=True)
    best = comp[~comp["Model"].str.startswith("Baseline")].iloc[0]["Model"]

    # best model for each horizon (1h, 2h, 3h): accuracy table + final refit on all data
    models, hrows, extra = {}, [], {}
    for h in HORIZONS:
        Xh, yh, tsh, rnh = build_short(df, all_lots, h)
        trh, teh = _time_split(tsh)
        m = _models()[best].fit(Xh[trh], yh[trh])
        pred = m.predict(Xh[teh])
        hrows.append({"Horizon": f"+{h} hour", "Baseline MAE": evaluate(yh[teh], rnh[teh])["MAE (pts %)"],
                      **evaluate(yh[teh], pred)})
        if h == 1:
            extra = {"importance": _importance(m, Xh[teh], yh[teh]),
                     "test_ts": tsh[teh].values, "test_actual": yh[teh].values,
                     "test_pred": np.clip(pred, 0, 1)}
        models[h] = _models()[best].fit(Xh, yh)
    return {"comparison": comp, "best_name": best, "models": models, "horizon_table": pd.DataFrame(hrows),
            "features": list(X.columns), **extra}


def train_plan(df, all_lots):
    X, y, ts, d = build_plan(df, all_lots)
    tr, te = _time_split(ts)
    # baseline: historical average by (lot, weekday, hour) from TRAIN period only
    key = d["lot_id"] + "|" + d["ts"].dt.dayofweek.astype(str) + "|" + d["ts"].dt.hour.astype(str)
    mean_map = y[tr].groupby(key[tr]).mean()
    base_pred = key[te].map(mean_map).fillna(y[tr].mean())
    rows = [{"Model": "Baseline (avg of same lot, weekday, hour)", **evaluate(y[te], base_pred)}]
    for name, m in _models().items():
        m.fit(X[tr], y[tr])
        rows.append({"Model": name, **evaluate(y[te], m.predict(X[te]))})
    comp = pd.DataFrame(rows).sort_values("MAE (pts %)").reset_index(drop=True)
    best = comp[~comp["Model"].str.startswith("Baseline")].iloc[0]["Model"]
    m_eval = _models()[best].fit(X[tr], y[tr])
    imp = _importance(m_eval, X[te], y[te])
    final = _models()[best].fit(X, y)
    return {"comparison": comp, "best_name": best, "model": final, "features": list(X.columns), "importance": imp}


# --------------------------------------------------------------------------
# 4. PREDICTION HELPERS
# --------------------------------------------------------------------------
def latest_state(df):
    """Most recent known occupancy for each lot (acts like the 'live sensor' feed)."""
    rows = []
    for lot, g in df.sort_values("ts").groupby("lot_id"):
        r = g["rate"].values
        if len(r) == 0:
            continue
        rate_l1 = r[-2] if len(r) >= 2 else r[-1]
        rate_l2 = r[-3] if len(r) >= 3 else r[-1]
        rate_l24 = r[-25] if len(r) >= 25 else r[0]
        rows.append({"lot_id": lot, "ts": g["ts"].iloc[-1], "capacity": g["capacity"].iloc[-1],
                     "occupied": int(g["occupied"].iloc[-1]), "rate_now": r[-1], "rate_l1": rate_l1,
                     "rate_l2": rate_l2, "rate_l24": rate_l24, "roll3": np.mean(r[-3:]) if len(r) >= 3 else r.mean()})
    return pd.DataFrame(rows)


def predict_short(short, df, all_lots, horizons=HORIZONS, rain=0.0, event=0, holiday=0):
    st_ = latest_state(df)
    out = []
    for h in horizons:
        t_ts = st_["ts"] + pd.Timedelta(hours=h)
        X = pd.concat([
            st_[["rate_now", "rate_l1", "rate_l2", "rate_l24", "roll3"]].reset_index(drop=True),
            _calendar(t_ts),
            pd.DataFrame({"is_holiday": holiday, "rain": rain, "event": event}, index=range(len(st_))),
            _lot_onehot(st_["lot_id"], all_lots)], axis=1)[short["features"]]
        p = np.clip(short["models"][h].predict(X), 0, 1)
        out.append(pd.DataFrame({"lot_id": st_["lot_id"], "horizon": h, "time": t_ts.values,
                                 "pred_rate": p, "capacity": st_["capacity"]}))
    r = pd.concat(out, ignore_index=True)
    r["pred_free"] = (r["capacity"] * (1 - r["pred_rate"])).round().astype(int)
    r["status"] = r["pred_rate"].apply(status_of)
    return r


def predict_plan(plan, lot_caps, times, rain=0.0, event=0, holiday=0):
    """lot_caps: dict lot_id -> capacity.  times: list of Timestamps. Returns long dataframe."""
    all_lots = [c[4:] for c in plan["features"] if c.startswith("lot_")]
    rows = [(l, t) for l in lot_caps for t in times]
    d = pd.DataFrame(rows, columns=["lot_id", "time"])
    X = pd.concat([_calendar(d["time"]),
                   pd.DataFrame({"is_holiday": holiday, "rain": rain, "event": event}, index=d.index),
                   _lot_onehot(d["lot_id"], all_lots)], axis=1)[plan["features"]]
    d["pred_rate"] = np.clip(plan["model"].predict(X), 0, 1)
    d["capacity"] = d["lot_id"].map(lot_caps)
    d["pred_free"] = (d["capacity"] * (1 - d["pred_rate"])).round().astype(int)
    d["status"] = d["pred_rate"].apply(status_of)
    return d


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def recommend(pred, meta, dest_lat=None, dest_lon=None):
    """Rank lots: enough free spots first, then nearest to destination (and less crowded)."""
    r = pred.merge(meta[["lot_id", "lat", "lon", "type"]], on="lot_id")
    if dest_lat is not None and r["lat"].notna().all():
        r["distance_km"] = haversine_km(dest_lat, dest_lon, r["lat"], r["lon"]).round(2)
        r["walk_min"] = (r["distance_km"] * 12).round().astype(int)      # ~5 km/h
    else:
        r["distance_km"], r["walk_min"] = 0.0, 0
    r["score"] = r["distance_km"] + 1.5 * r["pred_rate"]
    r.loc[r["pred_free"] < 5, "score"] += 100                           # practically full -> push down
    return r.sort_values("score").reset_index(drop=True)


def best_arrival(plan_hourly):
    """plan_hourly: predict_plan output for ONE lot over many hours -> least crowded hour."""
    return plan_hourly.loc[plan_hourly["pred_rate"].idxmin()]


def dynamic_price(rate, base=20.0):
    """Demand-based pricing: busier lot -> higher Rs/hour (also nudges people to quieter lots)."""
    mult = np.select([rate < 0.5, rate < 0.7, rate < 0.85], [0.8, 1.0, 1.25], default=1.5)
    return np.round(base * mult, 1)


# --------------------------------------------------------------------------
# 5. ANALYTICS
# --------------------------------------------------------------------------
def hourly_profile(df):
    d = df.assign(hour=df["ts"].dt.hour, off=(df["ts"].dt.dayofweek >= 5).astype(int))
    return d.pivot_table(index="hour", columns=["lot_id", "off"], values="rate", aggfunc="mean")


def condition_effect(df):
    """Average occupancy with / without rain and events (what changes demand?)."""
    d = df.copy()
    d["Rain"] = np.where(d["rain"] > 0, "Rain", "No rain")
    d["Event"] = np.where(d["event"] > 0, "Event", "No event")
    d["Holiday"] = np.where(d["is_holiday"] > 0, "Holiday", "Normal day")
    parts = []
    for col in ("Rain", "Event", "Holiday"):
        t = d.groupby(["lot_id", col])["rate"].mean().unstack() * 100
        t["Factor"] = col
        parts.append(t.reset_index())
    return parts


def cluster_lots(df, k=3):
    """KMeans on each lot's average 24-hour usage curve -> groups lots with similar behaviour."""
    prof = df.assign(hour=df["ts"].dt.hour).pivot_table(index="lot_id", columns="hour", values="rate", aggfunc="mean")
    k = min(k, len(prof))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(prof)) \
        if k > 1 else np.zeros(len(prof), dtype=int)
    wk = df.assign(off=df["ts"].dt.dayofweek >= 5).groupby(["lot_id", "off"])["rate"].mean().unstack()
    out = pd.DataFrame({"Peak hour": prof.idxmax(axis=1), "Avg occupancy %": (prof.mean(axis=1) * 100).round(1),
                        "Weekend/Weekday ratio": (wk[True] / wk[False]).round(2), "Cluster": labels}, index=prof.index)
    return out.reset_index()
