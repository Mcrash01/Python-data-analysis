import streamlit as st

def ml_models():
    st.title('ML MODELS')

    st.write('Here you can find the different ML models that we have trained and tested and the results of each one of them.')

    st.write('The models used are the following:')
    st.write('1. Random Forest')
    st.write('2. Random Forest tuned with GridSearchCV')
    st.write('3. SVM')
    st.write('4. SVM tuned with GridSearchCV')
    st.write('5. Linear Regression')
    st.write('6. Neural Network (scikit-learn)')
    st.write('7. Neural Network (TensorFlow)')
    st.write('8. XGBoost')

    st.write('The metric used to evaluate the models is the following: RMSE (Root Mean Squared Error).')

    st.write('The results of the models are the following:')

    st.image('src/img/models_eval.png')


