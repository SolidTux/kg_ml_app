#/bin/python3

#import the necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np 
from PIL import Image
from kost_NN import inference
import matplotlib.pyplot as plt


class plot_type:
    def __init__(self,data):
        self.data = data
        self.fig=None
        self.update_layout=None

    def bar(self,x,y,color):
        self.fig=px.bar(self.data,x=x,y=y,color=color)

    def pie(self,x,y):
        self.fig = px.pie(self.data,values=x,names=y)

        
    def set_title(self,title):
        
        self.fig.update_layout(
                title=f"{title}",
                    yaxis=dict(tickmode="linear"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=18))

    def set_title_x(self,title):
        
        self.fig.update_layout(
                title=f"{title}",
                    xaxis=dict(tickmode="linear"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=18))

    def set_title_pie(self,title):
        self.fig.update_layout(title=title,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=18))
        


    def plot(self):
        st.write(self.fig)

class slide_bar:
    value=4
    def __init__(self,title,x,y):
        self.title = title
        self.x=x
        self.y=y
        self.slide_bar = None
        

    def set(self):
        self.slide_bar = st.slider(self.title,self.x,self.y)
        slide_bar.value=self.slide_bar

class select_box:
    value="tyrion"
    def __init__(self,data):
        self.data=data
        self.box=None
    def place(self,title,key):
        header(title)
        self.box = st.selectbox(str(key),self.data)
        select_box.value=self.box

def title(text,size,color):
    st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',unsafe_allow_html=True)

def header(text):
    st.markdown(f"<p style='color:grey;'>{text}</p>",unsafe_allow_html=True)

# def gh_classification():

    # headers = ['Particle', 'lgE', 'X', 'Y', 'CoreDist', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']

    # energy = st.sidebar.slider('Energy [eV (log10)]', 0, 99, 20, 1)        # Energy [eV (log10)]	13 –  19	energy
    # x_core = st.sidebar.slider('X [m]', 0, 91, 80, 1)
    # y_core = st.sidebar.slider('Y [m]', 0, 91, 10, 1)
    # # core_dist = st.sidebar.slider('Core distance mm', 0, 91, 60, 1)      # Core distance [m]	0 –  91	core_distance
    # ze = st.sidebar.slider('Zenith [°]', 0, 60, 10, 1)                     # Zenith [°]	0 –  60	zenith 
    # Az = st.sidebar.slider('Azimuth [°]', 0, 360, 60, 1)                   # azimuth [°]	0 –  360	azimuth
    # ne = st.sidebar.slider('Electron number [log10]', 2.0, 8.7, 4.3, 0.1)  # Electron number [log10]	2 –  8.7	electron_number
    # nmu = st.sidebar.slider('Muon number [log10]', 2.0, 7.7, 3.0, 0.1)     # Muon number [log10]	2 –  7.7	muon_number
    # age = st.sidebar.slider('Shower Age', 0.1, 1.48,0.5, 0.01)             # Shower age	0.1 –  1.48	shower_age

    # row = [energy, x_core, y_core, ze, Az, ne, nmu, age]

    # #  RUN THE PIPELINES
    # if (st.button('Let\'s go!')):
    #     results = inference(row)
    #     for _model, _particle in results.items():
    #         st.write(f'Model {_model}: {_particle}')

#SET UP THE MAIN WINDOW
st.title('Machine learning particle classification using for KASCADE data')

# st.subheader('by Victoria Tokareva ([@Victoria.Tokareva](mailto:Victoria.Tokareva@kit.edu))')

st.markdown(
"""

<br><br/>
KASCADE was a very successful large detector array which recorded data during more than 20 years on site of the KIT-Campus North, Karlsruhe, Germany (formerly Forschungszentrum, Karlsruhe)
at 49,1°N, 8,4°E; 110m a.s.l. KASCADE collected within its lifetime more than 1.7 billion events of which some 433.000.000 survived all quality cuts and are made available here for public
usage via web portal <a href='https://kcdc.ikp.kit.edu/'>KCDC</a> (KASCADE Cosmic Ray Data Centre).

"""
, unsafe_allow_html=True)
image = Image.open('static/kascade_title.png')
image.thumbnail((295, 213),Image.ANTIALIAS)
st.image(image)
st.markdown(
"""
In this app you can compare predictions made by different machine learning methods on our preselected datasets.

"""
, unsafe_allow_html=True)
should_tell_me_more = False #st.button('Tell me more')
if should_tell_me_more:
    tell_me_more()
    st.markdown('---')
else:
    st.markdown('---')

    title("Work with datasets", 30, 'black')

    header('datasets')

    option_1_s = st.selectbox('',[1,2])

    title("Dataframe's structure", 24, 'black')
    #считать данные d dataframe
    if option_1_s == 1:
        df = pd.read_csv('./data/dataset1.csv')
    elif option_1_s == 2:
        df = pd.read_csv('./data/dataset2.csv')
    else:
        pass
    st.write(df.head(10))


    title("Dataset parameter distributions", 24, 'black')
    header('parameter')

    option_2_s = st.selectbox('', ['Energy', 'X_core', 'Y_core', 'Ze', 'Az', 'Ne', 'Nmu', 'Age'])
    col = 'lgE'
    d = {
        'Energy': 'lgE', 
        'X_core': 'X', 
        'Y_core': 'Y', 
        'Ze': 'Ze', 
        'Az': 'Az', 
        'Ne': 'lgNe', 
        'Nmu': 'lgNmu', 
        'Age': 'Age'
    }
    # par_names = ['lgE', 'X', 'Y', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']
    col = d[option_2_s ]
    hist_values = np.histogram(df[col], bins=50, range=(df[col].min(), df[col].max()))[0]
    st.bar_chart(hist_values)

    title("Neural network prediction", 24, 'black')
    header('model')
    option_3_s = st.selectbox('', ['QGSJet-4-based gamma-hadron classifier', 'QGSJet-4-based  mass composition classifier', 'Epos-LHC-based gamma-hadron classifier',\
         'Epos-LHC-based mass composition classifier', 'Sibyll-23c-based gamma-hadron classifier', 'Sibyll-23c-based mass composition classifier'])

    # pie2 = plot_type(t_data1)
    # pie2.pie("imp","season")
    # pie2.set_title_pie(stb2.value)
    # pie2.plot()
    mod = {
        'QGSJet-4-based gamma-hadron classifier': 'qgs-4_pr_gm', 
        'QGSJet-4-based  mass composition classifier': 'qgs-4_wo_gm_log', 
        'Epos-LHC-based gamma-hadron classifier': 'epos-LHC_pr_gm',
        'Epos-LHC-based mass composition classifier': 'epos-LHC_wo_gm_log', 
        'Sibyll-23c-based gamma-hadron classifier': 'sibyll-23c_pr_gm', 
        'Sibyll-23c-based mass composition classifier': 'sibyll-23c_wo_gm_log'
    }
    model = mod[option_3_s]

    for i in range(10):
        row = np.array(df.iloc[i])
        row
    results = new_inference(row, model)
    # for _model, _particle in results.items():
    #     st.write(f'Model {_model}: {_particle}')
    #     _model
  
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Gammas', 'Protons', 'Helium', 'Carbon', 'Silicon', 'Iron'
    sizes = [15, 30, 45, 10, 0, 15]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=80)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.legend(sizes, [labels])

    st.pyplot(fig1)

    # gh_classification()  #(df) = ?



