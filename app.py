#/bin/python3

#import the necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np 
from PIL import Image
from kost_NN import stat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def title(text,size,color):
    st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',unsafe_allow_html=True)

def header(text):
    st.markdown(f"<p style='color:grey;'>{text}</p>",unsafe_allow_html=True)

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

    st.write("""
        * **lgE** primary particle energy inducing the shower [𝑙𝑜𝑔10 eV]
        * **X**, **Y** shower core position (x, y) [m]
        * **Ze** zenith angle with respect to the vertical [degree]
        * **Az** azimuth angle with respect to north [degree]
        * **lgNe** number of electrons at observation level [𝑙𝑜𝑔10 number]
        * **lgNm** number of muons at observation level [𝑙𝑜𝑔10 number]
        * **Age** shower shape parameter
    """)
    #считать данные d dataframe
    if option_1_s == 1:
        df = pd.read_csv('./data/dataset1.csv')
    elif option_1_s == 2:
        df = pd.read_csv('./data/dataset2.csv')
    else:
        pass
    st.write(df.head(10))


    title("Dataset parameter distributions", 24, 'black')

    title("1d distributions", 16, 'black')
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

    plot_tit = {
        'Energy': 'Energy, eV', 
        'X_core': 'X, m', 
        'Y_core': 'Y, m', 
        'Ze': 'Zenith, deg', 
        'Az': 'Azimuth, deg', 
        'Ne': 'log Ne', 
        'Nmu': 'log Nmu', 
        'Age': 'Shower age'
    }

    col = d[option_2_s ]
    # hist_values = np.histogram(df[col], bins=50, range=(df[col].min(), df[col].max()))[0]
    # st.bar_chart(hist_values)
    fig0, ax0 = plt.subplots()
    ax0.hist(df[col], bins=50)
    ax0.set_xlabel(plot_tit[option_2_s])
    st.pyplot(fig0)

##################################################################################

    title("2d distributions", 16, 'black')

    header('parameters')

    option_2_s_2d = st.selectbox('', ['Shower core', 'Electron-muon distribution'])
    fig2, ax2 = plt.subplots()
    if option_2_s_2d == "Shower core":
        ax2.hist2d(df.X, df.Y, bins=100,  cmap=plt.cm.viridis)
        # ax2.set_title('Shower print')
        ax2.set_xlabel('X, meters')
        ax2.set_ylabel('Y, meters')
    elif option_2_s_2d == 'Electron-muon distribution':
        xbins = np.arange(df.lgNmu.min(), df.lgNmu.max(), 0.05) # muons
        ybins = np.arange(df.lgNe.min(), df.lgNe.max(), 0.05) # electrons
        plt.hist2d(df.lgNmu, df.lgNe, bins=[xbins,ybins], cmap = plt.cm.rainbow, norm=mcolors.LogNorm())
        cbar = plt.colorbar()
        ax2.set_xlabel("$\\rm{log}_{10}(N_{\\mu})$")
        ax2.set_ylabel("$\\rm{log}_{10}(N_{e})}$")
        # ax2.set_title("Electron-muon distribution")
    else:
        pass

    st.pyplot(fig2)

##################################################################################

    title("Neural network prediction", 24, 'black')
    header('model')
    option_3_s = st.selectbox('', ['QGSJet-4-based gamma-hadron classifier', 'QGSJet-4-based hadron mass composition classifier', 'Epos-LHC-based gamma-hadron classifier',\
         'Epos-LHC-based hadron mass composition classifier', 'Sibyll-23c-based gamma-hadron classifier', 'Sibyll-23c-based hadron mass composition classifier'])

    mod = {
        'QGSJet-4-based gamma-hadron classifier': 'qgs-4_pr_gm', 
        'QGSJet-4-based hadron mass composition classifier': 'qgs-4_wo_gm_log', 
        'Epos-LHC-based gamma-hadron classifier': 'epos-LHC_pr_gm',
        'Epos-LHC-based hadron mass composition classifier': 'epos-LHC_wo_gm_log', 
        'Sibyll-23c-based gamma-hadron classifier': 'sibyll-23c_pr_gm', 
        'Sibyll-23c-based hadron mass composition classifier': 'sibyll-23c_wo_gm_log'
    }
    model = mod[option_3_s]

    labels, sizes = stat(model, df)

    header('predicts that the selected dataset contains:')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=80)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.legend(sizes, [labels])

    st.pyplot(fig1)

    title("What's next?", 24, 'black')

    st.write('To learn how we trained our models, proceed to our publication [New insights from old cosmic rays: A novel analysis of archival KASCADE data](https://arxiv.org/abs/2108.03407).')
    # st.write('</br>')
    st.write('Developed by [V. Tokareva](https://www-kseta.ttp.kit.edu/fellows/Victoria.Tokareva/) for [Astroparticle Physics Research Group](https://research.jetbrains.org/groups/astroparticle-physics/)')

