import streamlit as st
from streamlit_folium import folium_static
from streamlit.components.v1 import html
    




def map():
    st.title('MAP')

    st.write('Fatality Count due to Landslides from 2007 to 2016')

    # HTML Folium Map
    with open("map.html", "r", encoding="utf-8") as file:
        map_html = file.read()
        st.components.v1.html(map_html, height=600)

    st.write("""
             The map shows the number of fatalities due to landslides from 2007 to 2016. 
             We can see that the number of fatalities is higher in Asia and South America (developping countries),
              even if the number of landslides is higher in Europe and North America (developped countries).
              This might be due to the fact that developped countries have better infrastructure and are
              more prepared to face natural disasters. Moreover the number of fatalities is higher
              in the mountainous regions such as Pakistan and China. Finally the density of population
              in mountainous regions is higher in thoose countries.""")

    
    

    



    

