import pandas as pd


def group_by_city(df):
    out = df.groupby('City').size().reset_index(name='Count')
    return out




def group_by_city_domain(df):
    return df.groupby(['City','Crime Domain']).size().reset_index(name='Count')