import streamlit as st

def homepage():
    st.title('HOMEPAGE')

    st.image('src/img/landslide.jpg', width=500)
    st.title('Context')
    st.write('Landslides are one of the most pervasive hazards in the world, causing more than 11,500 fatalities in 70 countries since 2007. Saturating the soil on vulnerable slopes, intense and prolonged rainfall is the most frequent landslide trigger.')
    st.title('Content')
    st.write('The Global Landslide Catalog (GLC) was developed with the goal of identifying rainfall-triggered landslide events around the world, regardless of size, impacts or location. The GLC considers all types of mass movements triggered by rainfall, which have been reported in the media, disaster databases, scientific reports, or other sources. The GLC has been compiled since 2007 at NASA Goddard Space Flight Center.')

    st.title('Dataset')
    st.write('The dataset is available at https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Export/dd9e-wu2v')
