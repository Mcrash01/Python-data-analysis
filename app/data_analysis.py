import streamlit as st

from exploration import preprocess_landslide_data,top_landslide_cnt,preprocessing_fatality_count

def data_analysis():
    st.title('DATA ANALYSIS')

    st.write('This is data analysis')
    number = st.slider('Select a number', min_value=0, max_value=10, value=0)
    landslides_data = preprocess_landslide_data()  # Use the preprocess_landslide_data function from earlier
    landslides_data = preprocessing_fatality_count(landslides_data)

    top_landslide_cnt(landslides_data,number)
    
