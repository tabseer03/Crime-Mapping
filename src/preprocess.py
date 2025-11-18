import pandas as pd


def parse_datetime(df, datetime_col='Time of Occurrence'):
    # convert column that holds full timestamp
    df['DateTime'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
    df['Hour'] = df['DateTime'].dt.hour
    df['Date'] = df['DateTime'].dt.date
    df['Weekday'] = df['DateTime'].dt.weekday
    # cleanup city
    df['City'] = df['City'].str.strip()
    return df