#/bin/python3

#import the necessary libraries
import streamlit as st
import joblib
#import pandas as pd
from PIL import Image
from kost_NN import inference
#import datetime

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
#@st.cache(allow_output_mutation=True)
#def load(scaler_path, model_path):
    #sc = joblib.load(scaler_path)
    #model = joblib.load(model_path)
    #return sc, model

def tell_me_more():
    st.button('Back to gamma hadron separation')  # will change state and hence trigger rerun and hence reset should_tell_me_more
    st.write('Here will be more text')

def gh_classification():
    #SET UP THE SIDEBAR
    headers = ['Particle', 'lgE', 'X', 'Y', 'CoreDist', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']
    # age =           st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
    # glucose =       st.sidebar.slider("Glucose Level", 0, 200, 25, 1)    #feature_name, min, max, default, step

   ## Range selector
    # cols1,_ = st.beta_columns((1,2)) # To make it narrower
    # format = 'MMM DD, YYYY'  # format output
    # start_date = dt.date(year=2021,month=1,day=1)-relativedelta(years=2)  #  I need some range in the past
    # end_date = dt.datetime.now().date()-relativedelta(years=2)
    # max_days = end_date-start_date

    # slider = cols1.slider('Select date', min_value=start_date, value=end_date ,max_value=end_date, format=format)
    # ## Sanity check
    # st.table(pd.DataFrame([[start_date, slider, end_date]],
    #                 columns=['start',
    #                         'selected',
    #                         'end'],
    #                 index=['date']))

    # today = datetime.date.today()
    # tomorrow = today + datetime.timedelta(days=1)
    # start_date = st.date_input('Start date', today)
    # end_date = st.date_input('End date', tomorrow)
    # if start_date < end_date:
    #     st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    # else:
    #     st.error('Error: End date must fall after start date.')

    energy = st.sidebar.slider('Energy [eV (log10)]', 0, 99, 20, 1)
    x_core = st.sidebar.slider('X [m]', 0, 91, 80, 1)
    y_core = st.sidebar.slider('Y [m]', 0, 91, 10, 1)
    # core_dist = st.sidebar.slider('Core distance mm', 0, 91, 60, 1)
    ze = st.sidebar.slider('Zenith [°]', 0, 60, 10, 1)
    Az = st.sidebar.slider('Azimuth [°]', 0, 360, 60, 1)
    ne = st.sidebar.slider('Electron number [log10]', 2.0, 8.7, 4.3, 0.1)
    nmu = st.sidebar.slider('Muon number [log10]', 2.0, 7.7, 3.0, 0.1)
    age = st.sidebar.slider('Shower Age', 0.1, 1.48,0.5, 0.01)

# zimuth [°]	0 –  360	azimuth
# Core distance [m]	0 –  91	core_distance
# Datetime	1998-05-08 16:35:38 –  2013-01-15 09:40:36	datetime
# Electron number [log10]	2 –  8.7	electron_number
# Energy [eV (log10)]	13 –  19	energy
# Muon number [log10]	2 –  7.7	muon_number
# Shower age	0.1 –  1.48	shower_age
# Zenith [°]	0 –  60	zenith

    row = [energy, x_core, y_core, ze, Az, ne, nmu, age]

    #  RUN THE PIPELINES
    if (st.button('Let\'s go!')):
        #feat_cols = ['lgE', 'X', 'Y', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']
        results = inference(row)
        #display the output
        for _model, _particle in results.items():
            st.write(f'Model {_model}: {_particle}')

#SET UP THE MAIN WINDOW
st.title('Machine lerning gamma hadron separation for KASCADE experiment')

st.subheader('by Victoria Tokareva ([@Victoria.Tokareva](mailto:Victoria.Tokareva@kit.edu))')

st.markdown(
"""

<br><br/>
KASCADE was a very successful large detector array which recorded data during more than 20 years on site of the KIT-Campus North, Karlsruhe, Germany (formerly Forschungszentrum, Karlsruhe)
at 49,1°N, 8,4°E; 110m a.s.l. KASCADE collected within its lifetime more than 1.7 billion events of which some 433.000.000 survived all quality cuts and are made available here for public
usage via web portal <a href='https://kcdc.ikp.kit.edu/'>KCDC</a> (KASCADE Cosmic Ray Data Centre).

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


