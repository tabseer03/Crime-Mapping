# app.py
from flask import Flask, render_template, request
from src.data_loader import load_data
from src.preprocess import parse_datetime
from src.aggregations import group_by_city
from src.geocode import city_coords
from src.viz import heatmap_from_grouped

app = Flask(__name__)

raw = load_data()
data = parse_datetime(raw)


time_windows = {
    "All": None,
    "Morning (05–12)": (5, 12),
    "Afternoon (12–17)": (12, 17),
    "Evening (17–22)": (17, 22),
    "Night (22–05)": (22, 5),
    "Late Night (22–02)": (22, 2)
}


def filter_time_window(df, start_hour, end_hour):
    """Return rows whose Hour is in the window.
    Handles windows that cross midnight (e.g. 22 -> 2)."""
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


# ------------------ Flask route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # default selections
    selected_time_label = "All"
    selected_gender = "All"
    selected_domain = "All"

    if request.method == "POST":
        selected_time_label = request.form.get("time_range", "All")
        selected_gender = request.form.get("gender", "All")
        selected_domain = request.form.get("domain", "All")

    # start from global preprocessed dataframe
    df = data.copy()

    # 1) Apply time filter
    tw = time_windows.get(selected_time_label)
    if tw is None:
        # 'All' — no time filtering
        df_time = df
    else:
        start, end = tw
        df_time = filter_time_window(df, start, end)

    # 2) Apply gender filter
    if selected_gender in ["M", "F"]:
        df_gender = df_time[df_time['Victim Gender'] == selected_gender]
    else:
        df_gender = df_time

    # 3) Apply crime domain filter
    if selected_domain != "All":
        df_filtered = df_gender[df_gender['Crime Domain'] == selected_domain]
    else:
        df_filtered = df_gender

    # 4) Group by city and add coords
    grouped = group_by_city(df_filtered)   # returns City + Count
    grouped = add_coordinates(grouped)

    # 5) Create heatmap and save to static file
    m = heatmap_from_grouped(grouped)
    m.save("static/map.html")

    # domain options come from data (include 'All' at top)
    domain_options = ["All"] + sorted(data['Crime Domain'].dropna().unique().tolist())
    print("\n=== FILTER DEBUG ===")
    print("Selected time:", selected_time_label)
    print("Selected gender:", selected_gender)
    print("Selected domain:", selected_domain)
    print("Rows after filtering:", len(df_filtered))
    print("Cities:", grouped['City'].tolist())

    return render_template(
    "index.html",
    time_options=list(time_windows.keys()),
    selected_time=selected_time_label,
    selected_gender=selected_gender,
    selected_domain=selected_domain,
    domain_options=domain_options,
    record_count=len(df_filtered),
    city_count=len(grouped)
)



if __name__ == "__main__":
    app.run(debug=True)
