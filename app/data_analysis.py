import streamlit as st

from exploration import preprocess_landslide_data,top_landslide_cnt,preprocessing_fatality_count

def data_analysis():
    st.title('DATA ANALYSIS')

    st.image('src/img/landslide.jpg')

    st.write('This is data analysis')
    
    st.write('This is data analysis')
    st.image('src/img/Landslides on World Map.png')

    st.write('This is data analysis')
    st.image('src/img/Total Number of Landslides by Month.png')
    
    st.write('This is data analysis')
    st.image('src/img/missing values by feature.png')

    st.write('This is data analysis')
    st.image('src/img/Total Number of Landslides by Year.png')

    st.write('This is data analysis')
    st.image('src/img/unique values.png')

    
    
    number = st.slider('Select a number', min_value=0, max_value=10, value=0)
    landslides_data = preprocess_landslide_data()  # Use the preprocess_landslide_data function from earlier
    landslides_data = preprocessing_fatality_count(landslides_data)

    top_landslide_cnt(landslides_data,number)
    
