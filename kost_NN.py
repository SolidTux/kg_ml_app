
import os
import numpy as np
from joblib import load
import streamlit as st

#-------------------------------------------------------------------------------------------
#                Pt.3. Interference
#-------------------------------------------------------------------------------------------

particles = {
    "wo_gm_log" : {0. : "proton", 1.4 : "He", 2.5 : "C", 3.3 : "Si", 4. : "Fe"},
    "pr_gm" : {0. : "gamma" , 1. : "proton"},
}
hadronic_models = [ "epos-LHC", "qgs-4", "sibyll-23c" ]

# extract models if not already there
if not os.path.isdir('kostunin_trees'):
    print('Extracting archived models...')
    os.system('tar -jxf kostunin_trees.tar.bz2')
# load models
regressor = {}
for pg in particles:
    for hm in hadronic_models:
        suff = hm + '_' + pg
        regressor[suff] = load(f'kostunin_trees/classifier/DecisionTreeRegressor_{suff}.joblib')

def inference(row):
    '''Returns the model predictions for particular data samples.

       Args:
          row: particular sample of data

      # Returns: the models prediction for the row in human readable format
    '''
    #feat_cols = ['lgE', 'X', 'Y', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']
    results = {}
    X = np.array([row])
    for pg in particles:
        for hm in hadronic_models:
            suff = hm + '_' + pg
            mass = np.round(regressor[suff].predict(X), decimals = 1)
            results[suff] = particles[pg][mass[0]]
    return results

#def test_inference():
    ##inference input data
    #row = a = np.random.rand(len(n_headers))
    #model = load( 'models/RandomForest.joblib')
    #scaler = load('models/StScaler.joblib')
    #print('Result: ', inference(row, scaler, model, n_headers))

#if __name__ == '__main__':
    #test_inference()
