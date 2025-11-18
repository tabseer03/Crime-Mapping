from flask import Flask, render_template, request
from src.data_loader import load_data
from src.preprocess import parse_datetime
from src.aggregations import group_by_city
from src.geocode import get_coords, city_coords
from src.viz import heatmap_from_grouped


app = Flask(__name__)


# Load & preprocess once at startup
raw = load_data()
data = parse_datetime(raw)


# add coords cached




def add_coords(df):
df['lat'] = df['City'].map(lambda c: city_coords.get(c, (None,None))[0])
df['lon'] = df['City'].map(lambda c: city_coords.get(c, (None,None))[1])
return df




@app.route('/', methods=['GET','POST'])
def index():
# read filters from form
time_range = request.form.get('time_range')
gender = request.form.get('gender')
domain = request.form.get('domain')


df = data.copy()
# apply filters (implement filter_time_window as in notebook)
# group
grouped = group_by_city(df)
grouped = add_coords(grouped)
m = heatmap_from_grouped(grouped)
m.save('static/map.html')
return render_template('index.html')


if __name__ == '__main__':
app.run(debug=True)