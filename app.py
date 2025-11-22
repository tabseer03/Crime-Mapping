from flask import Flask, render_template, request, jsonify
from src.data_loader import load_data
from src.preprocess import parse_datetime
from src.aggregations import group_by_city
from src.geocode import city_coords
from src.viz import heatmap_from_grouped

# ---------- FORECAST IMPORTS ----------
import numpy as np
import pandas as pd
from pmdarima import auto_arima
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# -----------------------------------------

app = Flask(__name__)

# Load dataset once
raw = load_data()
data = parse_datetime(raw)

# Time windows
time_windows = {
    "All": None,
    "Morning (05–12)": (5, 12),
    "Afternoon (12–17)": (12, 17),
    "Evening (17–22)": (17, 22),
    "Night (22–05)": (22, 5),
    "Late Night (22–02)": (22, 2)
}

def filter_time_window(df, start_hour, end_hour):
    if start_hour is None:
        return df
    if start_hour < end_hour:
        return df[(df['Hour'] >= start_hour) & (df['Hour'] < end_hour)]
    else:
        return df[(df['Hour'] >= start_hour) | (df['Hour'] < end_hour)]

def add_coordinates(df):
    df['lat'] = df['City'].map(lambda c: city_coords.get(c, (None, None))[0])
    df['lon'] = df['City'].map(lambda c: city_coords.get(c, (None, None))[1])
    return df

def police_to_severity(x):
    try:
        x = float(x)
    except:
        return "Medium"
    if x >= 13: return "High"
    elif x >= 6: return "Medium"
    return "Low"


# ======================================================
# MAIN PAGE
# ======================================================
@app.route("/", methods=["GET", "POST"])
def index():

    selected_time_label = request.form.get("time_range", "All")
    selected_gender = request.form.get("gender", "All")
    selected_domain = request.form.get("domain", "All")
    selected_city = request.form.get("city", None)

    df = data.copy()

    # 1) TIME FILTER
    tw = time_windows.get(selected_time_label)
    if tw:
        df = filter_time_window(df, tw[0], tw[1])

    # 2) GENDER FILTER
    if selected_gender in ["M", "F"]:
        df = df[df["Victim Gender"] == selected_gender]

    # 3) DOMAIN FILTER
    if selected_domain != "All":
        df = df[df["Crime Domain"] == selected_domain]

    # Add severity
    if "Police Deployed" in df.columns:
        df["Severity_Level"] = df["Police Deployed"].apply(police_to_severity)
    else:
        df["Severity_Level"] = "Medium"

    grouped = group_by_city(df)
    grouped = add_coordinates(grouped)

    # Generate heatmap
    m = heatmap_from_grouped(grouped)
    m.save("static/map.html")

    domain_options = ["All"] + sorted(data["Crime Domain"].dropna().unique().tolist())
    city_options = sorted(data["City"].unique().tolist())

    return render_template(
        "index.html",
        selected_time=selected_time_label,
        selected_gender=selected_gender,
        selected_domain=selected_domain,
        selected_city=selected_city,

        time_options=list(time_windows.keys()),
        domain_options=domain_options,
        city_options=city_options,

        record_count=len(df),
        city_count=len(grouped)
    )


# ======================================================
# AJAX MAP UPDATE ROUTE
# ======================================================
@app.route("/update_map", methods=["POST"])
def update_map():

    selected_time = request.form.get("time_range", "All")
    selected_gender = request.form.get("gender", "All")
    selected_domain = request.form.get("domain", "All")
    selected_city = request.form.get("city", None)

    df = data.copy()

    # Time filter
    tw = time_windows.get(selected_time)
    if tw:
        df = filter_time_window(df, tw[0], tw[1])

    # Gender filter
    if selected_gender in ["M", "F"]:
        df = df[df["Victim Gender"] == selected_gender]

    # Domain filter
    if selected_domain != "All":
        df = df[df["Crime Domain"] == selected_domain]

    # City filter (optional)
    if selected_city and selected_city != "All":
        df = df[df["City"] == selected_city]

    # Severity
    if "Police Deployed" in df.columns:
        df["Severity_Level"] = df["Police Deployed"].apply(police_to_severity)
    else:
        df["Severity_Level"] = "Medium"

    grouped = group_by_city(df)
    grouped = add_coordinates(grouped)

    # Update heatmap
    m = heatmap_from_grouped(grouped)
    m.save("static/map.html")

    return {
        "map_html": '<iframe src="/static/map.html?v=1" width="100%" height="430px"></iframe>'
    }


# ======================================================
# 7-DAY FORECAST ROUTE
# ======================================================
@app.route("/predict_crime", methods=["POST"])
def predict_crime():

    # preserve filters
    selected_time_label = request.form.get("time_range", "All")
    selected_gender = request.form.get("gender", "All")
    selected_domain = request.form.get("domain", "All")
    selected_city = request.form.get("city", None)

    df_full = data.copy()
    df_full["Date"] = pd.to_datetime(df_full["Date"])

    daily = df_full.groupby("Date").size().reset_index(name="Count").sort_values("Date")

    # Forecast logic
    if len(daily) < 10:
        avg = int(daily["Count"].mean()) if len(daily) > 0 else 0
        forecast = [avg] * 7
    else:
        try:
            model = auto_arima(daily["Count"], seasonal=False, suppress_warnings=True)
            forecast_raw = model.predict(n_periods=7)
            forecast = [int(max(0, round(float(x)))) for x in list(forecast_raw)]
        except:
            avg = int(daily["Count"].mean())
            forecast = [avg] * 7

    last_date = daily["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)

    forecast_table = [(future_dates[i].strftime("%Y-%m-%d"), forecast[i]) for i in range(7)]

    # Create Plot
    plt.figure(figsize=(8,4))
    plt.plot(daily["Date"], daily["Count"], label="Past")
    plt.plot(future_dates, forecast, marker="o", linestyle="--", label="Forecast")
    plt.title("Crime counts — historical + 7-day forecast")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    domain_options = ["All"] + sorted(data["Crime Domain"].dropna().unique().tolist())
    city_options = sorted(data["City"].unique().tolist())

    return render_template(
        "index.html",

        # preserve filters
        selected_time=selected_time_label,
        selected_gender=selected_gender,
        selected_domain=selected_domain,
        selected_city=selected_city,

        time_options=list(time_windows.keys()),
        domain_options=domain_options,
        city_options=city_options,

        forecast_graph=graph,
        forecast_table=forecast_table,
        record_count=len(data),
        city_count=len(data["City"].unique())
    )


# ======================================================
# CITY CRIME PREDICTION
# ======================================================
@app.route("/predict_city_crimes", methods=["POST"])
def predict_city_crimes():

    selected_time_label = request.form.get("time_range", "All")
    selected_gender = request.form.get("gender", "All")
    selected_domain = request.form.get("domain", "All")
    selected_city = request.form.get("city", None)

    df_city = data[data["City"] == selected_city]

    if df_city.empty:
        result = {"error": f"No data for {selected_city}"}
    else:
        top_descriptions = df_city["Crime Description"].value_counts().head(3).index.tolist()
        top_domains = df_city["Crime Domain"].value_counts().head(3).index.tolist()
        avg_pd = df_city["Police Deployed"].mean()
        predicted_severity = police_to_severity(avg_pd)

        result = {
            "city": selected_city,
            "likely_crimes": top_descriptions,
            "likely_domains": top_domains,
            "predicted_severity": predicted_severity,
            "confidence": "High" if len(df_city) > 50 else "Medium"
        }

    domain_options = ["All"] + sorted(data["Crime Domain"].dropna().unique().tolist())
    city_options = sorted(data["City"].unique().tolist())

    return render_template(
        "index.html",

        city_prediction=result,

        selected_time=selected_time_label,
        selected_gender=selected_gender,
        selected_domain=selected_domain,
        selected_city=selected_city,

        time_options=list(time_windows.keys()),
        domain_options=domain_options,
        city_options=city_options,

        record_count=len(data),
        city_count=len(data["City"].unique())
    )


# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
