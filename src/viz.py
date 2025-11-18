import folium
from folium.plugins import HeatMap, HeatMapWithTime
from .config import MAP_CENTER, MAP_ZOOM




def heatmap_from_grouped(df, center=MAP_CENTER, zoom=MAP_ZOOM, radius=25, blur=20):
    # df must have lat, lon, Count
    heat_data = df[['lat','lon','Count']].dropna().values.tolist()
    m = folium.Map(location=center, zoom_start=zoom)
    HeatMap(heat_data, radius=radius, blur=blur).add_to(m)
    return m




def heatmap_by_hour(df_grouped_by_hour, center=MAP_CENTER, zoom=MAP_ZOOM):
    # df_grouped_by_hour: list of dataframes or list of lists [[lat,lon,weight], ...] per hour
    m = folium.Map(location=center, zoom_start=zoom)
    HeatMapWithTime(df_grouped_by_hour, index=[f"{h}:00" for h in range(len(df_grouped_by_hour))]).add_to(m)
    return m