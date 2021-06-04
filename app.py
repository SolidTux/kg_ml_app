#/bin/python3

#import the necessary libraries
import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from toy_NN import inference

def my_notes():
    st.markdown(
    '''
    Вопрос: как много гамма-кандидатов?

    Что можно менять: 
        - дата каты (много)
        - модель машинного обучения (+классич. анализ)
        - модель симуляций?

    Что возвращаем? 
        - % гамма кандидатов (на основе каскадовских данных)
        - сколько всего данных, соотв таким катам, находится среди КАСКАДОВСКИХ данных?
        - картинка с балансом гамм и не-гамм
        - картинка, показывающая, какая часть каскадовских данных соответствовала заявленным катам
        - насколько мы уверены в своей оценке (+ кнопка с объяснением, как эта оценка считается) - возможно, confusion matrix или roc
    '''
    )

# optimized model and scaler load using streamlit cache
@st.cache(allow_output_mutation=True)
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc , model

def tell_me_more():
    st.button('Back to gamma hadron separation')  # will change state and hence trigger rerun and hence reset should_tell_me_more
    st.write('Here will be more text')
    
def gh_classification():
    #SET UP THE SIDEBAR
    headers = ['Particle', 'lgE', 'X', 'Y', 'CoreDist', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age'] 
    # age =           st.sidebar.number_input("Age in Years", 1, 150, 25, 1)  
    # glucose =       st.sidebar.slider("Glucose Level", 0, 200, 25, 1)    #feature_name, min, max, default, step

    energy = st.sidebar.slider('lgE', 0, 99, 20, 1)
    x_core = st.sidebar.slider('X', 0, 99, 20, 1)
    y_core = st.sidebar.slider('Y', 0, 99, 20, 1)
    core_dist = st.sidebar.slider('CoreDist', 0, 99, 20, 1)
    ze = st.sidebar.slider('Ze', 0, 99, 20, 1)
    Az = st.sidebar.slider('Az', 0, 99, 20, 1)
    ne = st.sidebar.slider('lgNe', 0, 99, 20, 1)
    nmu = st.sidebar.slider('lgNmu', 0, 99, 20, 1)
    age = st.sidebar.slider('Age', 0, 99, 20, 1)

    row = [energy, x_core, y_core, core_dist, ze, Az, ne, nmu, age]

    #  RUN THE PIPELINES
    if (st.button('Let\'s go!')):
        feat_cols = ['lgE', 'X', 'Y', 'CoreDist', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']

        sc, model = load('models/StScaler.joblib', 'models/RandomForest.joblib')
        result = inference(row, sc, model, feat_cols)
        
        #display the output 
        st.write(result) 
    

#SET UP THE MAIN WINDOW
st.title('Machine lerning gamma hadron separation for KASCADE experiment')

st.subheader('by Victoria Tokareva ([@Victoria.Tokareva](mailto:Victoria.Tokareva@kit.edu))')

st.markdown(
"""

<br><br/>

KASCADE is... 

"""
, unsafe_allow_html=True)
image = Image.open('static/kascade_title.png')
st.image(image, use_column_width=True)
st.markdown(
"""
Explore the predictions made by different machine learning methods using the filters on the left. Do you agree with the model?

To read more about the methods, click below.

"""
, unsafe_allow_html=True)
should_tell_me_more = st.button('Tell me more')
if should_tell_me_more:
    tell_me_more()
    st.markdown('---')
else:
    st.markdown('---')
    gh_classification()  #(df) = ?
        
#st.write('Please fill in the details of the person under consideration in the left sidebar and click on the button below!')


