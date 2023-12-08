import streamlit as st
from streamlit_folium import folium_static
from streamlit.components.v1 import html
    




def map():
    st.title('MAP')

    st.write('This is map')

    # Intégrer la carte HTML dans Streamlit
    with open("map.html", "r", encoding="utf-8") as file:
        map_html = file.read()
        st.components.v1.html(map_html, height=600)

    
    

    



    

