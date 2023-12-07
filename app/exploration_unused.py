

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import datetime
import folium
from folium import plugins



landslides = pd.read_csv(r"C:\Users\victo\Documents\ESILV\Git\Python-data-analysis\data\Global_Landslide_Catalog_Export.csv", sep=';')
landslides[["event_date2", "event_hour", "eventAmOrPm"]]= landslides["event_date"].str.split(expand= True)
landslides[["event_hour2", "event_minute", "event_second"]]= landslides["event_hour"].str.split(':',expand= True)
landslides[["event_date_day", "event_date_month", "event_date_year"]]= landslides["event_date2"].str.split('/',expand= True)
landslides["event_hour2"]= landslides["event_hour2"].astype(int)
landslides["event_date"]= pd.to_datetime(landslides["event_date"])
landslides.loc[landslides['eventAmOrPm'] == 'PM', 'event_hour2'] = landslides.loc[landslides['eventAmOrPm'] == 'PM', 'event_hour2'] + 12




def preprocess_landslide_data():
    # Read the CSV file
    landslides = pd.read_csv(r"C:\Users\victo\Documents\ESILV\Git\Python-data-analysis\data\Global_Landslide_Catalog_Export.csv", sep=';')

    # Perform data preprocessing
    landslides[["event_date2", "event_hour", "eventAmOrPm"]] = landslides["event_date"].str.split(expand=True)
    landslides[["event_hour2", "event_minute", "event_second"]] = landslides["event_hour"].str.split(':', expand=True)
    landslides[["event_date_day", "event_date_month", "event_date_year"]] = landslides["event_date2"].str.split('/', expand=True)
    landslides["event_hour2"] = landslides["event_hour2"].astype(int)
    landslides["event_date"] = pd.to_datetime(landslides["event_date"])
    landslides.loc[landslides['eventAmOrPm'] == 'PM', 'event_hour2'] += 12

    return landslides



# missing values
missing_values = landslides.isnull().sum() + landslides.applymap(lambda x: (str)(x).strip() == 'unknown').sum() 

plt.figure(figsize=(20, 10))
ax = sns.barplot(x=missing_values.index, y=missing_values, palette="rocket") #order=missing_values.sort_values(ascending=False).index
# for i in ax.containers:
#     ax.bar_label(i,)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Missing values')
plt.title('Missing values by feature')
plt.tight_layout()
plt.show()

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def display_missing_values_plot(data):
    # Calculate missing values
    missing_values = data.isnull().sum() + data.applymap(lambda x: str(x).strip() == 'unknown').sum()

    # Create the missing values plot
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=missing_values.index, y=missing_values, palette="rocket")
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Missing values')
    plt.title('Missing values by feature')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)



# ## Dataset Cleaning
# 

# ### Time Columns
# 

# In[69]:


# landslides[["event_date2", "event_hour", "eventAmOrPm"]]= landslides["event_date"].str.split(expand= True)
#landslides["event_hour"]=datetime(landslides["event_hour"])


# In[70]:


landslides[["event_hour2", "event_minute", "event_second"]]= landslides["event_hour"].str.split(':',expand= True)


# In[71]:


landslides[["event_date_month", "event_date_day", "event_date_year"]]= landslides["event_date2"].str.split('/',expand= True)
# landslides[["event_time2"]]= landslides["event_date"].str.split('/',expand= True).concat()


# In[72]:


landslides["event_hour2"]= landslides["event_hour2"].astype(int)
landslides["event_date"]= pd.to_datetime(landslides["event_date"])


# In[73]:


landslides.loc[landslides['eventAmOrPm'] == 'PM', 'event_hour2'] = landslides.loc[landslides['eventAmOrPm'] == 'PM', 'event_hour2'] + 12


# ### Unique values
# 

# In[74]:


unique_values =  landslides.nunique()

plt.figure(figsize=(20, 10))
ax = sns.barplot(x=unique_values.index, y=unique_values, palette="rocket") #order=missing_values.sort_values(ascending=False).index
# for i in ax.containers:
#     ax.bar_label(i,)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('unique values')
plt.title('unique values')


# In[75]:


non_unique_values =  len(landslides)-landslides.nunique()

plt.figure(figsize=(20, 10))
ax = sns.barplot(x=non_unique_values.index, y=non_unique_values, palette="rocket") #order=missing_values.sort_values(ascending=False).index
# for i in ax.containers:
#     ax.bar_label(i,)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('non unique values')
plt.title(' non unique values')


# ### Duplicate rows
# 

# In[76]:


duplicateRowsDF = landslides[landslides.duplicated()]
duplicateRowsDF


# In[77]:


duplicateRowsDF['event_date']=duplicateRowsDF['event_date'].astype(str)
duplicateRowsDF.dtypes


# In[78]:


duplicateRowsDF = landslides[landslides.duplicated(landslides[["event_hour","event_minute","country_name","landslide_category","event_date_day","event_date_month","event_date_year"]])]
duplicateRowsDF=duplicateRowsDF.sort_values(by="event_date_year")
duplicateRowsDF


# In[79]:


duplicateRowsDF = landslides[landslides.duplicated(landslides[["country_name", "landslide_category", "event_date_day", "event_date_month", "event_date_year","country_name","location_description"]])]#,"country_name","landslide_category","location_description"
duplicateRowsDF=duplicateRowsDF.sort_values(by="location_description")
duplicateRowsDF


# In[80]:


#landslides = landslides.drop_duplicates(subset=["event_hour", "event_minute", "country_name", "landslide_category", "event_date_day", "event_date_month", "event_date_year","country_name","location_description"])


# ## Data Visualization
# 

# ### Lattitude / Longitude Landslide distribution
# 

# In[81]:



landslides = landslides[~landslides['fatality_count'].isin(['', 'unknown'])]
landslides = landslides.dropna(subset=['fatality_count'])
landslides['fatality_count'] = landslides['fatality_count'].str.replace(',', '')
landslides['fatality_count'] = landslides['fatality_count'].astype(int)
landslides['admin_division_population'] = landslides['admin_division_population'].str.replace(',', '')
landslides = landslides.dropna(subset=['admin_division_population'])
landslides['admin_division_population'] = landslides['admin_division_population'].astype(int)


# In[82]:



location_data = landslides[['latitude', 'longitude','fatality_count','landslide_size']]

# Resetting index after removing rows
location_data.reset_index(drop=True, inplace=True)

location_data['fatality_count'] = location_data['fatality_count'].astype(int)  # If 'fatality_count' is represented as strings


# In[83]:


sns.violinplot(landslides['latitude'],palette="rocket")


# In[84]:


sns.violinplot(landslides['longitude'],palette="rocket")


# ### Fatal Count by Landslide on Map
# 

# In[85]:


# Drop columns with longitude values > 180 or < -180 in the 'landslides' DataFrame
landslides = landslides.loc[(landslides['longitude'] <= 180) & (landslides['longitude'] >= -180)]


# In[86]:



import cartopy.crs as ccrs
import cartopy.feature as cfeature

location_data_sorted = location_data.sort_values(by='fatality_count', ascending=True)
# Assuming you have longitude, latitude columns in your 'landslides' DataFrame
# Replace 'longitude' and 'latitude' with your actual column names
longitude = landslides['longitude']
latitude = landslides['latitude']

# Create a figure and axis with Cartopy projection
plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot world map
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN)
#ax.add_feature(cfeature.LAND, edgecolor='black')

# Plot scatter points on the map
#ax.scatter(longitude, latitude, transform=ccrs.PlateCarree(), s=5, color='blue', label='Landslides')
scatter=sns.scatterplot(x='longitude', y='latitude',hue='landslide_size', data=location_data_sorted, palette='Reds', size='fatality_count',sizes=(50, 5000))

# Add legend
plt.legend(labelspacing=0.0)



# Get the current axes and figure
ax = plt.gca()
fig = plt.gcf()

# Get the handles and labels for the first legend (colors)
handles, labels = scatter.get_legend_handles_labels()

# Create the first legend for colors
first_legend = fig.legend(handles=handles[:8], labels=labels[:8],bbox_to_anchor=(0.0, 1.0),loc='upper right')



# Create the second legend for marker sizes
second_legend = fig.legend(handles=handles[8:], labels=labels[8:], loc='lower left',ncol= len(handles[8:]),columnspacing=3.0,handletextpad=3.0,bbox_to_anchor=(0.0, 0.0))


# Add the legends to the plot
ax.legend_.remove()
ax.add_artist(first_legend)
ax.add_artist(second_legend)

# Show the plot
plt.title('Landslides on World Map')
plt.tight_layout



# Set larger marker sizes if needed

plt.show()


# ### Temporal Seasonality
# 

# In[101]:


month_counts = landslides['event_date_month'].value_counts().reset_index()
month_counts.columns = ['event_date_month', 'count']
month_counts["event_date_month"] = month_counts["event_date_month"].astype(int)
month_counts.sort_values(by='event_date_month', ascending=True)

month_dictionnary = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July", 8:"August",9:"September",10:"October",11:"November",12:"December"}

# Plotting
plt.figure(figsize=(20, 10))
ax = sns.barplot(x=month_counts['event_date_month'].map(month_dictionnary), y=month_counts['count'], palette="rocket", order=month_dictionnary.values())
plt.xlabel('Month')
plt.ylabel('Landslide') 
plt.title('Total Number of Landslides by Month')
plt.show()


# In[88]:


year_counts = landslides['event_date_year'].value_counts().reset_index()
year_counts.columns = ['event_date_year', 'count']
year_counts["event_date_year"] = year_counts["event_date_year"].astype(int)
year_counts.sort_values(by='event_date_year', ascending=True)

# Plotting
plt.figure(figsize=(20, 10))
sns.barplot(data=year_counts, x='event_date_year', y='count',palette="rocket")
plt.xticks(rotation=45)
plt.xlabel('year')
plt.ylabel('Lansdlide')
plt.title('Total Number of Landslides by Year')
plt.show()


# ### Landslide / country distribution
# 

# In[89]:


fatal_df = landslides.loc[(landslides['fatality_count'] > 0)]
fatal_df = fatal_df['country_name'].value_counts().reset_index()
fatal_df.columns = ['country_name', 'count']
fatal_df.sort_values(by='count', ascending=True)
fatal_df= fatal_df[:20]

# Plotting
plt.figure(figsize=(20, 10))
sns.barplot(data=fatal_df, x='country_name', y='count',palette="rocket")
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Fatal Landslide')
plt.title('Fatal Landslide by Country')
plt.show()


# In[90]:



country_landslides = landslides['country_name'].value_counts().reset_index()
country_landslides.columns = ['country_name', 'count']
country_landslides.sort_values(by='count', ascending=True)
country_landslides= country_landslides[:20]

# Plotting
plt.figure(figsize=(20, 10))
sns.barplot(data=country_landslides, x='country_name', y='count',palette="rocket")
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Landslide')
plt.title('Reported Landslide by Country')
plt.show()


# In[91]:


country_landslides = landslides.groupby('country_name')['fatality_count'].sum()
country_landslides = country_landslides.sort_values(ascending=False)[:20]  

# Plotting
plt.figure(figsize=(20, 10))
sns.barplot(country_landslides,palette="rocket")
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Fatality Count')
plt.title('Reported Landslide Fatality Count by Country')
plt.show()


# In[92]:



m=folium.Map(tiles="OpenStreetMap",max_bounds=True, zoom_start=2,location=[0.0, 0.0])
# plot heatmap
m.add_children(plugins.HeatMap(landslides[['latitude','longitude','fatality_count']], radius=15))
# for index, row in location_data.iterrows():
#     popup_info = f"Fatality Count: {row['fatality_count']}"
#     folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=row['fatality_count']/1000, popup=popup_info, fill=True).add_to(m)

m

