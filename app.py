"""
app.py - Smart Parking Availability & Prediction System (Streamlit dashboard)
Run:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import core

st.set_page_config(page_title="Smart Parking AI", page_icon="🅿️", layout="wide")
sns.set_theme(style="whitegrid")
ICON = {"Available": "🟢", "Filling up": "🟡", "Almost full": "🔴"}
COLOR = {"Available": "#2ca02c", "Filling up": "#ff9f1c", "Almost full": "#d62728"}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🅿️ Settings")
src = st.sidebar.radio("Data source", ["Demo data (synthetic)", "Upload CSV"])
df = meta = None
if src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV: timestamp, lot_id, occupied (+ optional capacity, rain, event, lat, lon)",
                                  type=["csv", "txt"])
    if up is not None:
        try:
            df, meta = core.load_csv(up)
        except Exception as e:
            st.sidebar.error(str(e))
if df is None:
    if src == "Upload CSV":
        st.sidebar.info("Showing demo data until a CSV is uploaded.")
    df, meta = core.generate_demo_data()

st.sidebar.subheader("Conditions (for prediction)")
rain_on = st.sidebar.checkbox("🌧️ Rain expected", value=False)
event_on = st.sidebar.checkbox("🎉 Event / sale / festival", value=False)
holiday_on = st.sidebar.checkbox("📅 Holiday", value=False)
RAIN, EVENT, HOLIDAY = (5.0 if rain_on else 0.0), int(event_on), int(holiday_on)
base_price = st.sidebar.number_input("Base parking price (Rs/hour)", value=20.0, step=5.0)

core.save_to_sqlite(df, "parking.db")          # SQL storage of history
all_lots = list(meta["lot_id"])
caps = dict(zip(meta["lot_id"], meta["capacity"]))


# ------------------------------------------------------------------ pipeline (cached)
@st.cache_data(show_spinner="Training models (first run may take 20-30 sec)...")
def run_pipeline(data: pd.DataFrame, lots: tuple):
    return core.train_short(data, list(lots)), core.train_plan(data, list(lots))

short, plan = run_pipeline(df, tuple(all_lots))
now_ts = df["ts"].max()
live = core.predict_short(short, df, all_lots, rain=RAIN, event=EVENT, holiday=HOLIDAY)
state = core.latest_state(df)

st.title("🅿️ Smart Parking Availability & Prediction System")
st.caption(f"Data: {df['ts'].min():%d %b %Y} → {now_ts:%d %b %Y %H:%M}  |  {len(all_lots)} parking lots  |  "
           f"{len(df):,} hourly records  |  'Now' = last data point ({now_ts:%d %b, %I %p})")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📍 Live Status", "🔮 Future Prediction", "🧭 Smart Recommendation", "📊 Analytics", "🧪 Model Report"])

# ------------------------------------------------------------------ TAB 1: LIVE
with tab1:
    st.subheader("How many spaces are free right now?")
    cols = st.columns(len(state))
    for col, (_, r) in zip(cols, state.iterrows()):
        s = core.status_of(r["rate_now"])
        free = int(r["capacity"] - r["occupied"])
        col.metric(f"{ICON[s]} {r['lot_id']}", f"{free} free", f"{r['rate_now']*100:.0f}% full", delta_color="off")
        col.progress(float(min(r["rate_now"], 1.0)))

    st.subheader("Availability forecast for the next 1-3 hours")
    piv = live.pivot(index="lot_id", columns="horizon", values="pred_free")
    pst = live.pivot(index="lot_id", columns="horizon", values="status")
    table = pd.DataFrame({f"+{h} hr": [f"{ICON[pst.loc[l, h]]} {piv.loc[l, h]} free" for l in piv.index]
                          for h in core.HORIZONS}, index=piv.index)
    table.insert(0, "Now", [f"{ICON[core.status_of(state.set_index('lot_id').loc[l, 'rate_now'])]} "
                             f"{int(state.set_index('lot_id').loc[l, 'capacity'] - state.set_index('lot_id').loc[l, 'occupied'])} free"
                             for l in piv.index])
    st.dataframe(table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for lot in all_lots:
        pts = [state.set_index("lot_id").loc[lot, "rate_now"] * 100] + \
              list(live[live["lot_id"] == lot].sort_values("horizon")["pred_rate"] * 100)
        ax.plot([0, 1, 2, 3], pts, marker="o", label=lot)
    ax.axhline(90, color="red", ls="--", lw=1); ax.set_xticks([0, 1, 2, 3], ["Now", "+1h", "+2h", "+3h"])
    ax.set_ylabel("Occupancy %"); ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    st.pyplot(fig); plt.close(fig)
    st.caption("Short-term model: current occupancy + recent hourly trend + weather/event signals.")

# ------------------------------------------------------------------ TAB 2: PLANNING
with tab2:
    st.subheader("Plan for any day or date")
    c1, c2 = st.columns(2)
    lot = c1.selectbox("Parking lot", all_lots)
    day = c2.date_input("Date", value=(now_ts + pd.Timedelta(days=1)).date())
    hrs = st.slider("Time window", 0, 23, (6, 22))
    times = [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in range(hrs[0], hrs[1] + 1)]
    pp = core.predict_plan(plan, {lot: caps[lot]}, times, rain=RAIN, event=EVENT, holiday=HOLIDAY)
    best = core.best_arrival(pp)
    worst = pp.loc[pp["pred_rate"].idxmax()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Least crowded time", f"{best['time']:%I %p}", f"{best['pred_free']} spots free", delta_color="off")
    m2.metric("Busiest time", f"{worst['time']:%I %p}", f"{worst['pred_rate']*100:.0f}% full", delta_color="off")
    pick = st.select_slider("When would you like to arrive?", options=list(pp["time"]),
                            format_func=lambda t: f"{t:%I %p}", value=pp["time"].iloc[len(pp) // 2])
    pr = pp[pp["time"] == pick].iloc[0]
    m3.metric(f"{pick:%I %p}", f"{ICON[pr['status']]} {pr['status']}",
              f"{pr['pred_free']} free | Rs {core.dynamic_price(pr['pred_rate'], base_price)}/hr", delta_color="off")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(pp["time"].dt.strftime("%H:%M"), pp["pred_rate"] * 100, color=[COLOR[s] for s in pp["status"]])
    ax.axhline(90, color="red", ls="--", lw=1); ax.set_ylabel("Predicted occupancy %")
    ax.tick_params(axis="x", rotation=60)
    st.pyplot(fig); plt.close(fig)
    if lot and best["pred_rate"] < 0.7:
        st.success(f"Tip: visit {lot} around **{best['time']:%I %p}** for the most available spaces.")
    st.caption("The planning model predicts using only the calendar (hour, weekday, holiday), weather, and events, "
               "so it works for any future date.")

# ------------------------------------------------------------------ TAB 3: RECOMMEND
with tab3:
    st.subheader("Which parking lot is best?")
    h = st.radio("How many hours from now will you arrive?", list(core.HORIZONS), format_func=lambda x: f"{x} hours later",
                 horizontal=True)
    has_geo = meta["lat"].notna().all()
    dest_lat = dest_lon = None
    if has_geo:
        dest_name = st.selectbox("Landmark near your destination", all_lots)
        dm = meta.set_index("lot_id").loc[dest_name]
        dest_lat, dest_lon = float(dm["lat"]), float(dm["lon"])
    else:
        st.info("No lat/lon data was found in the CSV, so ranking is based only on availability.")
    rec = core.recommend(live[live["horizon"] == h], meta, dest_lat, dest_lon)
    top = rec.iloc[0]
    if top["pred_free"] >= 5:
        msg = (f"👉 **{top['lot_id']}** is the best option: ~{top['pred_free']} spots free expected "
               f"({top['status']})")
        if has_geo:
            msg += f", {top['distance_km']} km from the destination (~{top['walk_min']} min walk)"
        st.success(msg + f". Price: Rs {core.dynamic_price(top['pred_rate'], base_price)}/hr")
    else:
        st.error("At that time, most lots appear nearly full. Try a little earlier or later.")
    show = rec.assign(Status=rec["status"].map(lambda s: f"{ICON[s]} {s}"),
                      **{"Free spots": rec["pred_free"], "Occupancy %": (rec["pred_rate"] * 100).round(0),
                         "Rs/hr": core.dynamic_price(rec["pred_rate"].values, base_price)})
    cols_show = ["lot_id", "Status", "Free spots", "Occupancy %", "Rs/hr"] + (["distance_km", "walk_min"] if has_geo else [])
    st.dataframe(show[cols_show].rename(columns={"lot_id": "Parking lot"}), use_container_width=True)
    st.caption("Ranking = first by available spaces, then by proximity to the destination and lower crowding. "
               "Dynamic pricing: the busier the lot, the higher the rate.")

# ------------------------------------------------------------------ TAB 4: ANALYTICS
with tab4:
    prof = core.hourly_profile(df)
    st.subheader("Hourly pattern: weekday vs weekend")
    sel = st.selectbox("Lot", all_lots, key="an_lot")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(prof.index, prof[(sel, 0)] * 100, label="Weekday", marker="o")
    ax.plot(prof.index, prof[(sel, 1)] * 100, label="Weekend", marker="o")
    ax.set_xlabel("Hour"); ax.set_ylabel("Avg occupancy %"); ax.legend()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Heatmap: day × hour")
    hm = df[df["lot_id"] == sel].assign(h=lambda d: d["ts"].dt.hour, d=lambda d: d["ts"].dt.dayofweek) \
        .pivot_table(index="d", columns="h", values="rate", aggfunc="mean") * 100
    hm.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(hm)]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    sns.heatmap(hm, cmap="YlOrRd", ax=ax, cbar_kws={"label": "% full"})
    st.pyplot(fig); plt.close(fig)

    st.subheader("Impact of rain / event / holiday (avg occupancy %)")
    eff = core.condition_effect(df)
    cc = st.columns(3)
    for c, t in zip(cc, eff):
        c.caption(t["Factor"].iloc[0])
        c.dataframe(t.drop(columns="Factor").round(1).set_index("lot_id"), use_container_width=True)

    st.subheader("Lots with similar behavior (KMeans clustering)")
    st.dataframe(core.cluster_lots(df).set_index("lot_id"), use_container_width=True)

# ------------------------------------------------------------------ TAB 5: MODEL REPORT
with tab5:
    st.subheader("1) Short-term model (+1 hour) - comparison")
    st.dataframe(short["comparison"], use_container_width=True)
    st.caption("Time-based split: first 80% of days for training, last 20% for testing (no data leakage). "
               "MAE 'pts %' = average error in occupancy %. Baseline = 'same as it is now'.")
    st.subheader("2) Horizon-wise accuracy")
    st.dataframe(short["horizon_table"], use_container_width=True)
    st.subheader("3) Planning model (any future date) - comparison")
    st.dataframe(plan["comparison"], use_container_width=True)
    st.caption("Baseline = same lot, same weekday, same hour historical average.")

    c1, c2 = st.columns(2)
    for c, title, imp in ((c1, "Short-term: what matters most", short["importance"]),
                          (c2, "Planning: what matters most", plan["importance"])):
        with c:
            st.caption(title)
            top_imp = imp.head(8)[::-1]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.barh(top_imp.index, top_imp.values, color="C4")
            st.pyplot(fig); plt.close(fig)

    st.subheader("Actual vs Predicted (+1h, test period, last 5 days)")
    td = pd.DataFrame({"ts": short["test_ts"], "actual": short["test_actual"], "pred": short["test_pred"]})
    g = td.groupby("ts").mean().tail(24 * 5)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(g.index, g["actual"] * 100, label="Actual"); ax.plot(g.index, g["pred"] * 100, label="Predicted", alpha=0.85)
    ax.set_ylabel("Avg occupancy % (all lots)"); ax.legend()
    st.pyplot(fig); plt.close(fig)
