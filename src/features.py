def filter_time_window(df, start_hour, end_hour):
    if start_hour < end_hour:
        return df[(df['Hour'] >= start_hour) & (df['Hour'] < end_hour)]
    else:
        return df[(df['Hour'] >= start_hour) | (df['Hour'] < end_hour)]
