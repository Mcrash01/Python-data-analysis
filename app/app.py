import pandas as pd
import streamlit as st
from homepage import *
from documentation import *
from data_analysis import *
from map import *
from ml_models import *

st.title('Landslide')


display_page = st.sidebar.selectbox('Quelle page souhaitez-vous visiter ?',('HOMEPAGE', 'DATA ANALYSIS', 'MAP', 'ML MODELS', 'DOCUMENTATION'))
# HOMEPAGE STUFF
if display_page == 'HOMEPAGE':
    homepage()
    if st.button('Load Data'):
        df_train = pd.read_csv('./data/Data_Train.csv')
        st.write("Data Successfully loaded")
        st.dataframe(data=df_train)
        st.line_chart(data=df_train, x='Airline', y ='Price')
# DATA ANALYSIS STUFF
elif display_page == 'DATA ANALYSIS':
    data_analysis()
# MAP STUFF
elif display_page == 'MAP':
    map()
# ML MODELS STUFF
elif display_page == 'ML MODELS':
    ml_models()
# DOCUMENTATION STUFF
elif display_page == 'DOCUMENTATION':
    documentation()


