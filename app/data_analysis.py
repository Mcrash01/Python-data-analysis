import streamlit as st

from exploration import preprocess_landslide_data,top_landslide_cnt,preprocessing_fatality_count

def data_analysis():
    st.title('DATA ANALYSIS')
    
    st.write(
    """
    ### Landslide Distribution on World Map

    The map visualizes global landslides, using color gradients to denote the severity, ranging from minor to catastrophic incidents. Dot sizes on the map are proportional to the number of fatalities caused by each landslide event.

    An evident observation is the prevalence of catastrophic landslides in Asia, often leading to a significant loss of life. Additionally, there are instances of smaller-scale landslides that, despite their size, resulted in considerable fatalities.
    """
    )
    st.image('src/img/Landslides on World Map.png')



    st.write(
    """
    ### Seasonal Trends in Landslides
    The graph depicting total landslides per month reveals distinct seasonal trends. Initial surges occur during early and late winter, coinciding with freezing and thawing phases respectively. However, the most notable escalation emerges in summer, correlating with high temperatures. This pronounced increase is explainned due not only to temperature fluctuations but also to the interplay between dry land conditions and subsequent heavy rainfall, a prevalent occurrence during this season. The combination of these factors significantly amplifies landslide activity. This pattern underscores the impact of temperature variations and moisture dynamics on landslide occurrences throughout the year, emphasizing the vulnerability of landscapes to these seasonal influences.
    """
    )
    st.image('src/img/Total Number of Landslides by Month.png')
    
    st.write(
    """
    ### Missing Data Observations
    
    The analysis of missing values across features reveals crucial insights:
    
    - **Complete Data:** Essential attributes such as longitude, latitude, source name, and date exhibit no missing values, ensuring robustness in these critical data fields.
    - **Mostly Complete Data:** Other metrics, including fatality count, gazetteer distance, landslide size, and event descriptions, also demonstrate high completeness, providing substantial information for analysis.
    - **Significant Missing Data:** However, two features, namely landslide settings but most importantly injury count, display a notable amount of missing data. These gaps in information might impact the comprehensive understanding of these specific attributes and necessitate further scrutiny or imputation strategies to mitigate the missing values' impact on analysis.
    
    """
    )
    st.image('src/img/missing values by feature.png')

    st.write(
    """
    ### Long-Term Landslide Trends
    
    The graph depicting the total number of landslides by year illustrates a distinct pattern. Initially, there are relatively few reported landslides. However, a significant linear increase is evident from 2007 onwards. This steady rise denotes a consistent uptrend in reported landslide occurrences.
    
    The years 2010 and 2011 stand out prominently, displaying a sharp deviation from this linear progression with a notably higher number of reported landslides. These two years notably depart from the otherwise consistent linear trend observed in landslide occurrences over time.
    """
    )
    st.image('src/img/Total Number of Landslides by Year.png')

    st.write(
    """
    ### Different Values in Features
    
    The image shows the uniqueness of values found in the data. Some feature have lots of different values, while others have only a few. It gives an idea of how varied the information is across different parts of the dataset.
    """
    )
    st.image('src/img/unique values.png')

    
    

    landslides_data = preprocess_landslide_data()  # Use the preprocess_landslide_data function from earlier
    landslides_data = preprocessing_fatality_count(landslides_data)



    st.write("""
    ### Landslides per Country by Fatalities

    This graph displays the number of landslides per country based on an adjustable fatality count. When including non-fatal landslides, the United States reports the most cases. However, filtering for fatal incidents reveals that India and China have higher reported cases where lives were lost.
    """)
    number = st.slider('Select a number', min_value=0, max_value=10, value=0)


    top_landslide_cnt(landslides_data, number)

    
