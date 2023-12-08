
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# import sklearn
# import warnings
# import datetime
# import folium
# from folium import plugins

def preprocess_landslide_data():
    # Read the CSV file
    landslides = pd.read_csv("data/Global_Landslide_Catalog_Export.csv", sep=';')

    # Perform data preprocessing
    landslides[["event_date2", "event_hour", "eventAmOrPm"]] = landslides["event_date"].str.split(expand=True)
    landslides[["event_hour2", "event_minute", "event_second"]] = landslides["event_hour"].str.split(':', expand=True)
    landslides[["event_date_day", "event_date_month", "event_date_year"]] = landslides["event_date2"].str.split('/', expand=True)
    landslides["event_hour2"] = landslides["event_hour2"].astype(int)
    landslides["event_date"] = pd.to_datetime(landslides["event_date"])
    landslides.loc[landslides['eventAmOrPm'] == 'PM', 'event_hour2'] += 12

    return landslides

def preprocessing_fatality_count(landslides):
    landslides = landslides[~landslides['fatality_count'].isin(['', 'unknown'])]
    landslides = landslides.dropna(subset=['fatality_count'])
    landslides['fatality_count'] = landslides['fatality_count'].str.replace(',', '')
    landslides['fatality_count'] = landslides['fatality_count'].astype(int)
    return landslides

def top_landslide_cnt(landslides,cnt):
 
    fatal_df = landslides.loc[(landslides['fatality_count'] >= cnt)]
    fatal_df = fatal_df['country_name'].value_counts().reset_index()
    fatal_df.columns = ['country_name', 'count']
    fatal_df.sort_values(by='count', ascending=True)
    fatal_df= fatal_df[:6]

    # Plotting
    fontsize=40
    plt.figure(figsize=(30, 20))
    sns.barplot(data=fatal_df, x='country_name', y='count',palette="rocket")
    plt.xticks(rotation=45,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Country',fontsize=fontsize)
    plt.ylabel('Landslide Number',fontsize=fontsize)
    plt.title('Landslide with at least ' + str(cnt) + ' fatality by Country',fontsize=fontsize)
    st.pyplot(plt)